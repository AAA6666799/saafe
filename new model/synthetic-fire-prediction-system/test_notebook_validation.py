import json
import sys

def validate_notebook(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Notebook is valid JSON")
        print(f"✅ Number of cells: {len(data['cells'])}")
        print(f"✅ Notebook format: {data['nbformat']}.{data['nbformat_minor']}")
        
        # Check for code cells
        code_cells = [cell for cell in data['cells'] if cell['cell_type'] == 'code']
        markdown_cells = [cell for cell in data['cells'] if cell['cell_type'] == 'markdown']
        print(f"✅ Code cells: {len(code_cells)}")
        print(f"✅ Markdown cells: {len(markdown_cells)}")
        
        # Check if last cell is markdown (should be key takeaways)
        if data['cells'][-1]['cell_type'] == 'markdown':
            print("✅ Last cell is markdown (key takeaways)")
            
        return True
    except Exception as e:
        print(f"❌ Error validating notebook: {e}")
        return False

if __name__ == "__main__":
    file_path = "notebooks/flir_scd41_unified_training_diagnostics.ipynb"
    success = validate_notebook(file_path)
    sys.exit(0 if success else 1)