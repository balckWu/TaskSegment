"""Collect only the reference source files needed to inspect a preflight mismatch."""
from pathlib import Path
import argparse
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iris2d.protocol import reference_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='iris2d_server_reference.zip')
    args = parser.parse_args()
    report = reference_report()
    print('Python:', report['python'].split()[0])
    for file, value in report['files'].items():
        print(value['status'].upper() + ': ' + file)
    print('Mismatches:', ', '.join(report['mismatches']) or 'none')
    relative_files = sorted({name.split('::')[0] for name in report['files']})
    relative_files += ['configs/paper_3seeds.json', 'configs/iris2d_baseline.json',
                       'configs/iris2d_reference.json', 'iris2d/protocol.py']
    output = Path(args.output).resolve()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('diagnostic_report.json', json.dumps(report, ensure_ascii=False, indent=2))
        for relative in relative_files:
            file = ROOT / relative
            if file.is_file():
                archive.write(file, relative)
    print('Saved reference code snapshot:', output)
    print('This archive contains reference code/config only; no images, weights or training results.')


if __name__ == '__main__':
    main()
