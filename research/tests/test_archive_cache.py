"""Offline transport recovery must never make network requests or hide gaps."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fetch_archive_roles import RangeFile


class ArchiveCache(unittest.TestCase):
    def test_offline_read_across_cached_blocks(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            (root/'source.json').write_text(json.dumps({'url':'fixture','bytes':12,'chunk_bytes':8}))
            (root/'00000000.bin').write_bytes(b'abcdefgh')
            (root/'00000001.bin').write_bytes(b'ijkl')
            with patch('fetch_archive_roles.requests.head',side_effect=AssertionError('network used')):
                source=RangeFile('fixture',root,chunk=8,offline=True)
                source.seek(6)
                self.assertEqual(source.read(6),b'ghijkl')

    def test_offline_missing_block_is_rejected(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            (root/'source.json').write_text(json.dumps({'url':'fixture','bytes':8,'chunk_bytes':8}))
            with patch('fetch_archive_roles.requests.head',side_effect=AssertionError('network used')):
                source=RangeFile('fixture',root,chunk=8,offline=True)
                with self.assertRaises(FileNotFoundError):source.read(1)


if __name__=='__main__':unittest.main()
