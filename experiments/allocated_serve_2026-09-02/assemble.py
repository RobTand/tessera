"""Substitute the section files into the receipt."""
import pathlib
import subprocess

RECEIPT = pathlib.Path('/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6/'
                       'docs/measurements/tessera-allocated-served-2026-09-02.md')
SECTIONS = {
    'KL_TABLE_PLACEHOLDER': 'sec5_kl.md',
    'PLACEHOLDER_SURROGATE': 'sec6.md',
    'PLACEHOLDER_SEPARATOR': 'sec7.md',
    'PLACEHOLDER_REFERENCES': 'sec8.md',
    'PLACEHOLDER_LIMITS': 'sec9.md',
    'PLACEHOLDER_REPRO': 'sec_repro.md',
}
text = RECEIPT.read_text()
for marker, name in SECTIONS.items():
    path = pathlib.Path('/home/rob/tmp/alloc-plans') / name
    if marker in text and path.exists():
        text = text.replace(marker, path.read_text().rstrip('\n'))
        print(f'  filled {marker} from {name}')
commit = subprocess.run(['git', '-C', str(RECEIPT.parents[2]), 'rev-parse', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()
if commit:
    text = text.replace('PLACEHOLDER_COMMIT', commit[:7])
RECEIPT.write_text(text)
left = [m for m in list(SECTIONS) + ['PLACEHOLDER_COMMIT'] if m in text]
print('still unfilled:', left or 'none')
