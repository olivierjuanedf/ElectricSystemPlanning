import json

with open('data_crunch.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

with open('notebook_structure.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total cells: {len(nb['cells'])}\n\n")
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        cell_id = cell.get('id', 'N/A')
        source_lines = cell.get('source', [])
        source_preview = "".join(source_lines[:3]).strip().replace('\n', ' ')
        
        # If markdown, let's capture the title
        title = ""
        if cell_type == 'markdown':
            for line in source_lines:
                if line.strip().startswith('#'):
                    title = line.strip()
                    break
        
        out.write(f"Cell {i} | Type: {cell_type} | ID: {cell_id} | Title: {title}\n")
        out.write(f"  Preview: {source_preview[:120]}\n\n")

print("Notebook structure successfully written to notebook_structure.txt!")
