#!/usr/bin/env python3
"""
输出项目/实验目录结构，并生成便于排查训练状态的文本报告。

特点：
1. 不读取文件内容，只读取文件名、大小、修改时间等元数据。
2. 不跟随符号链接，避免递归进入其他磁盘或形成循环。
3. 可限制最大深度和每个目录显示的条目数。
4. 自动汇总 checkpoint、日志、配置、结果文件。
5. 权限不足时不会中断，会在报告中标记。

示例：
    python print_server_structure.py /path/to/project \
        --max-depth 6 \
        --max-items 120 \
        --show-size \
        --show-mtime \
        --output server_structure.txt

若当前终端已经位于项目根目录：
    python print_server_structure.py . --max-depth 6 --show-size --show-mtime
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_IGNORE_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
}

TRAINING_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".engine",
    ".pkl",
    ".pickle",
}

CONFIG_SUFFIXES = {
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
}

LOG_SUFFIXES = {
    ".log",
    ".txt",
    ".csv",
    ".tsv",
    ".jsonl",
    ".out",
    ".err",
}

RESULT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".npy",
    ".npz",
    ".mat",
    ".xlsx",
}

IMPORTANT_NAMES = {
    "readme.md",
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "dockerfile",
    "train.py",
    "test.py",
    "eval.py",
    "evaluate.py",
    "inference.py",
    "infer.py",
}


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def safe_stat(path: Path):
    try:
        return path.lstat()
    except (OSError, PermissionError):
        return None


def should_ignore(path: Path, ignore_names: set[str], include_hidden: bool) -> bool:
    name = path.name
    if name in ignore_names:
        return True
    if not include_hidden and name.startswith("."):
        return True
    return False


def sorted_entries(directory: Path) -> tuple[list[Path], str | None]:
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        return [], "[无权限访问]"
    except OSError as exc:
        return [], f"[读取失败: {exc}]"

    def key(item: Path):
        try:
            is_dir = item.is_dir() and not item.is_symlink()
        except OSError:
            is_dir = False
        return (not is_dir, item.name.lower())

    entries.sort(key=key)
    return entries, None


class ReportState:
    def __init__(self) -> None:
        self.total_dirs = 0
        self.total_files = 0
        self.total_symlinks = 0
        self.total_bytes = 0
        self.extension_counts: Counter[str] = Counter()
        self.extension_bytes: Counter[str] = Counter()
        self.categories: dict[str, list[tuple[Path, int, float]]] = defaultdict(list)
        self.errors: list[str] = []

    def register_file(self, path: Path, size: int, mtime: float) -> None:
        self.total_files += 1
        self.total_bytes += size

        suffix = path.suffix.lower() or "[无扩展名]"
        self.extension_counts[suffix] += 1
        self.extension_bytes[suffix] += size

        lower_name = path.name.lower()
        lower_parts = {part.lower() for part in path.parts}

        if path.suffix.lower() in TRAINING_SUFFIXES:
            self.categories["模型与 checkpoint"].append((path, size, mtime))

        if path.suffix.lower() in CONFIG_SUFFIXES or lower_name in IMPORTANT_NAMES:
            self.categories["配置与入口文件"].append((path, size, mtime))

        if (
            path.suffix.lower() in LOG_SUFFIXES
            or "log" in lower_name
            or "logs" in lower_parts
            or "tensorboard" in lower_parts
        ):
            self.categories["日志与指标文件"].append((path, size, mtime))

        if (
            path.suffix.lower() in RESULT_SUFFIXES
            and (
                {"result", "results", "prediction", "predictions", "output", "outputs"}
                & lower_parts
            )
        ):
            self.categories["结果与预测文件"].append((path, size, mtime))


def describe_entry(
    path: Path,
    show_size: bool,
    show_mtime: bool,
) -> tuple[str, bool, bool]:
    """
    返回：
        描述文本,
        是否为普通目录,
        是否为符号链接
    """
    stat = safe_stat(path)
    if stat is None:
        return f"{path.name} [无法读取元数据]", False, False

    is_link = path.is_symlink()
    try:
        is_dir = path.is_dir() and not is_link
    except OSError:
        is_dir = False

    if is_link:
        try:
            target = os.readlink(path)
        except OSError:
            target = "?"
        text = f"{path.name} -> {target} [symlink]"
    elif is_dir:
        text = f"{path.name}/"
    else:
        text = path.name

    metadata = []
    if show_size and not is_dir:
        metadata.append(human_size(stat.st_size))
    if show_mtime:
        metadata.append(format_mtime(stat.st_mtime))

    if metadata:
        text += "  [" + " | ".join(metadata) + "]"

    return text, is_dir, is_link


def build_tree(
    root: Path,
    max_depth: int,
    max_items: int,
    ignore_names: set[str],
    include_hidden: bool,
    show_size: bool,
    show_mtime: bool,
    state: ReportState,
) -> list[str]:
    lines: list[str] = []

    root_stat = safe_stat(root)
    root_meta = []
    if root_stat is not None and show_mtime:
        root_meta.append(format_mtime(root_stat.st_mtime))

    root_title = str(root.resolve())
    if root_meta:
        root_title += "  [" + " | ".join(root_meta) + "]"
    lines.append(root_title)

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            entries, error = sorted_entries(directory)
            visible = [
                p for p in entries
                if not should_ignore(p, ignore_names, include_hidden)
            ]
            if error:
                lines.append(prefix + "└── " + error)
            elif visible:
                lines.append(
                    prefix
                    + f"└── … [达到最大深度，下面还有 {len(visible)} 个条目]"
                )
            return

        entries, error = sorted_entries(directory)
        if error:
            lines.append(prefix + "└── " + error)
            state.errors.append(f"{directory}: {error}")
            return

        visible_entries = [
            item
            for item in entries
            if not should_ignore(item, ignore_names, include_hidden)
        ]

        shown_entries = visible_entries[:max_items]
        omitted = len(visible_entries) - len(shown_entries)

        for index, item in enumerate(shown_entries):
            has_more_after = index < len(shown_entries) - 1 or omitted > 0
            branch = "├── " if has_more_after else "└── "
            child_prefix = prefix + ("│   " if has_more_after else "    ")

            text, is_dir, is_link = describe_entry(
                item,
                show_size=show_size,
                show_mtime=show_mtime,
            )
            lines.append(prefix + branch + text)

            stat = safe_stat(item)
            if is_link:
                state.total_symlinks += 1
            elif is_dir:
                state.total_dirs += 1
                walk(item, child_prefix, depth + 1)
            elif stat is not None:
                state.register_file(item, stat.st_size, stat.st_mtime)

        if omitted > 0:
            lines.append(
                prefix
                + f"└── … [本目录还有 {omitted} 个条目未显示；"
                  f"可增大 --max-items]"
            )

    if root.is_dir():
        walk(root, "", 0)
    else:
        stat = safe_stat(root)
        if stat is not None:
            state.register_file(root, stat.st_size, stat.st_mtime)

    return lines


def category_report(
    root: Path,
    state: ReportState,
    newest_limit: int,
) -> list[str]:
    lines = [
        "",
        "=" * 88,
        "训练相关文件汇总",
        "=" * 88,
    ]

    category_order = [
        "模型与 checkpoint",
        "配置与入口文件",
        "日志与指标文件",
        "结果与预测文件",
    ]

    for category in category_order:
        items = state.categories.get(category, [])
        lines.append("")
        lines.append(f"[{category}] 共 {len(items)} 个")

        if not items:
            lines.append("  未发现")
            continue

        # 最近修改的文件优先，便于判断当前训练状态。
        newest = sorted(items, key=lambda x: x[2], reverse=True)[:newest_limit]
        for path, size, mtime in newest:
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = path
            lines.append(
                f"  - {relative} | {human_size(size)} | {format_mtime(mtime)}"
            )

        if len(items) > newest_limit:
            lines.append(
                f"  … 仅显示最近修改的 {newest_limit} 个，"
                f"另有 {len(items) - newest_limit} 个未列出"
            )

    return lines


def extension_report(state: ReportState, limit: int = 30) -> list[str]:
    lines = [
        "",
        "=" * 88,
        "总体统计",
        "=" * 88,
        f"目录数: {state.total_dirs}",
        f"文件数: {state.total_files}",
        f"符号链接数: {state.total_symlinks}",
        f"已扫描文件总大小: {human_size(state.total_bytes)}",
        "",
        f"文件类型统计（按数量取前 {limit} 项）:",
    ]

    for suffix, count in state.extension_counts.most_common(limit):
        total_size = state.extension_bytes[suffix]
        lines.append(
            f"  {suffix:<16} {count:>8} 个  {human_size(total_size):>12}"
        )

    if state.errors:
        lines.extend(
            [
                "",
                "扫描错误或权限问题:",
                *[f"  - {item}" for item in state.errors],
            ]
        )

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输出服务器项目目录结构和训练文件汇总。"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="需要扫描的项目根目录，默认是当前目录。",
    )
    parser.add_argument(
        "--output",
        default="server_structure.txt",
        help="报告输出路径，默认 server_structure.txt。",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="最大递归深度，默认 6。",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=120,
        help="每个目录最多显示多少个条目，默认 120。",
    )
    parser.add_argument(
        "--newest-limit",
        type=int,
        default=80,
        help="每类训练文件最多列出多少个最近修改项，默认 80。",
    )
    parser.add_argument(
        "--show-size",
        action="store_true",
        help="显示文件大小。",
    )
    parser.add_argument(
        "--show-mtime",
        action="store_true",
        help="显示最后修改时间。",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="包括隐藏文件和隐藏目录。",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help=(
            "额外忽略的文件或目录名，可重复指定。"
            "例如：--ignore images --ignore masks"
        ),
    )
    parser.add_argument(
        "--no-default-ignore",
        action="store_true",
        help="不使用脚本内置的忽略目录列表。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"错误：路径不存在：{root}", file=sys.stderr)
        return 2

    if args.max_depth < 0:
        print("错误：--max-depth 不能小于 0。", file=sys.stderr)
        return 2

    if args.max_items <= 0:
        print("错误：--max-items 必须大于 0。", file=sys.stderr)
        return 2

    ignore_names = set(args.ignore)
    if not args.no_default_ignore:
        ignore_names |= DEFAULT_IGNORE_NAMES

    state = ReportState()

    header = [
        "服务器项目结构报告",
        "=" * 88,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"扫描根目录: {root.resolve()}",
        f"最大深度: {args.max_depth}",
        f"每目录最大条目数: {args.max_items}",
        f"包括隐藏项: {args.include_hidden}",
        f"忽略名称: {', '.join(sorted(ignore_names)) if ignore_names else '无'}",
        "=" * 88,
        "",
    ]

    tree_lines = build_tree(
        root=root,
        max_depth=args.max_depth,
        max_items=args.max_items,
        ignore_names=ignore_names,
        include_hidden=args.include_hidden,
        show_size=args.show_size,
        show_mtime=args.show_mtime,
        state=state,
    )

    report_lines = (
        header
        + tree_lines
        + category_report(root.resolve(), state, args.newest_limit)
        + extension_report(state)
    )

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"错误：无法写入报告 {output}: {exc}", file=sys.stderr)
        return 1

    print(f"报告已生成：{output}")
    print(
        f"扫描到 {state.total_dirs} 个目录、{state.total_files} 个文件、"
        f"{state.total_symlinks} 个符号链接。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
