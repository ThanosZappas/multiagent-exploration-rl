import numpy as np
import re
import os

def sort_references_inplace(filepath):
    """
    Sorts references in a text file directly, based on the release date.

    Parameters:
    filepath (str): The path to the text file to be sorted.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return

    def extract_year(reference_string):
        """Extracts the first 4-digit year from the reference string."""
        match = re.search(r'\b(18|19|20)\d{2}\b', reference_string)
        if match:
            return int(match.group(0))
        return 0

    # Sort the references
    sorted_references = sorted(references, key=extract_year, reverse=True)

    # Write the sorted references back to the same file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for ref in sorted_references:
                f.write(ref + '\n')
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")

def main():
    # # Create a dummy references file for the example
    # references_list = [
    #     "(Hart et al., 1968) P. E. Hart, N. J. Nilsson and B. Raphael, \"A Formal Basis for the Heuristic Determination of Minimum Cost Paths,\" in IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100-107, July 1968, doi: 10.1109/TSSC.1968.300136.",
    #     "(Yamauchi, 1997) B. Yamauchi, \"A frontier-based approach for autonomous exploration,\" Proceedings 1997 IEEE International Symposium on Computational Intelligence in Robotics and Automation CIRA'97. 'Towards New Computational Principles for Robotics and Automation', Monterey, CA, USA, 1997, pp. 146-151, doi: 10.1109/CIRA.1997.613851.",
    #     "(Bourgault et al., 2002)  Bourgault, F., Makarenko, A., Williams, S. B., Grocholsky, B., & Durrant-Whyte, H. F. (2002, September). Information based adaptive robotic exploration. In IEEE/RSJ International Conference on Intelligent Robots and Systems (Vol. 1, pp. 540–545). IEEE. https://doi.org/10.1109/IRDS.2002.1041410",
    #     "(Tai et al., 2016) Tai, L., & Liu, M. (2016). A robot exploration strategy based on Q-learning network. In IEEE International Conference on Real-time Computing and Robotics (RCAR), 2016. DOI: 10.1109/RCAR.2016.7784001"
    # ]
    
    # filepath = "references.txt"
    # with open(filepath, "w", encoding="utf-8") as f:
    #     for ref in references_list:
    #         f.write(ref + "\n")
            
    # # Sort the file directly
    # print(f"--- Sorting file: {filepath} ---")
    # sort_references_inplace(filepath)
    
    # # Verify the result by reading the file again
    # print(f"\n--- Content of {filepath} after sorting ---")
    # with open(filepath, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         print(line.strip())
        
    # # Clean up the dummy file
    # os.remove(filepath)
    references_file = "gym_env/common/references.txt"  # Replace with your actual file path
    sort_references_inplace(references_file)
if __name__ == "__main__":
    main()