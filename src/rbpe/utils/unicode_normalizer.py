import pickle
import re
import os

class UnicodeNormalizer:
    def __init__(self):
        """
        Initialize the normalizer with a unicode map.
        
        Args:
            unicode_map_path (str): Path to the unicode map pickle file
        """
        current_file = os.path.abspath(__file__)
        utils_dir = os.path.dirname(current_file)
        rbpe_dir = os.path.dirname(utils_dir)
        src_dir = os.path.dirname(rbpe_dir)
        rbpe_repo_dir = os.path.dirname(src_dir)
        self.unicode_map_path = os.path.join(rbpe_repo_dir, "inputs", "arabic_chars_rbpe_norm.pickle")
        self.unicode_map = self._load_unicode_map(self.unicode_map_path)
    
    def _load_unicode_map(self, map_path: str) -> dict:
        """Load the unicode mapping from a pickle file."""
        try:
            with open(map_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not load unicode map from {map_path}: {str(e)}")
    
    def normalize(self, text: str) -> str:
        """
        Normalize text according to the unicode normalization rules.
        Only replaces characters that are part of a multi-character token.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Split text into whitespace and non-whitespace sequences
        pattern = r'(\s+|\S+)'
        tokens = re.findall(pattern, text)
        result = []
        
        for token in tokens:
            # If it's whitespace or single character, keep as is
            if token.isspace() or len(token) == 1:
                result.append(token)
                continue
                
            # For multi-character tokens, apply normalization
            normalized_token = []
            for char in token:
                unicode_point = ord(char)
                if unicode_point not in self.unicode_map:
                    normalized_token.append(char)
                    continue
                    
                replacement = self.unicode_map[unicode_point]
                try:
                    if isinstance(replacement, list):
                        # Handle list of replacement points
                        replacement_chars = [chr(int(rep)) for rep in replacement]
                        normalized_token.append(''.join(replacement_chars))
                    else:
                        # Handle single replacement point
                        normalized_token.append(chr(int(replacement)))
                except (ValueError, OverflowError):
                    normalized_token.append(char)
            
            result.append(''.join(normalized_token))
        
        return ''.join(result)


if __name__ == "__main__":
    unicode_normalizer = UnicodeNormalizer()
    print(unicode_normalizer.normalize("حٲ"))
