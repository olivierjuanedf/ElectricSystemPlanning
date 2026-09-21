import json
import re

def create_markdown_cell(cell_id, text_lines):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in text_lines]
    }

def get_section_from_heading(line):
    line = line.strip()
    if re.match(r"^#\s+1\s*-", line) or line == "# 1 - Imports": return "1"
    if re.match(r"^#\s+2\s*-", line) or "Choice of your study case" in line: return "2"
    if re.match(r"^#\s+3\s*-", line) or line == "# 3 - Power Demand": return "3"
    if re.match(r"^##\s+3\.1", line): return "3.1"
    if re.match(r"^##\s+3\.2", line): return "3.2"
    if re.match(r"^###\s+3\.2\.1", line): return "3.2.1"
    if re.match(r"^###\s+3\.2\.2", line): return "3.2.2"
    if re.match(r"^###\s+3\.2\.3", line): return "3.2.3"
    if re.match(r"^##\s+3\.3", line): return "3.3"
    if re.match(r"^###\s+3\.3\.1", line): return "3.3.1"
    if re.match(r"^###\s+3\.3\.2", line): return "3.3.2"
    if re.match(r"^###\s+3\.3\.3", line): return "3.3.3"
    if re.match(r"^##\s+3\.4", line) or "Risk management and unsupplied energy" in line: return "3.4"
    if re.match(r"^#\s+4\s*-", line) or line == "# 4 - Generation": return "4"
    if re.match(r"^##\s+4\.1", line): return "4.1"
    if re.match(r"^##\s+4\.2\s*-", line): return "4.2"
    if re.match(r"^##\s+4\.2\.1", line): return "4.2.1"
    if re.match(r"^###\s+4\.2\.2", line): return "4.2.2"
    if re.match(r"^###\s+4\.2\.3", line): return "4.2.3"
    if re.match(r"^##\s+4\.3", line): return "4.3"
    if re.match(r"^###\s+4\.3\.1", line): return "4.3.1"
    if re.match(r"^###\s+4\.3\.2", line): return "4.3.2"
    if re.match(r"^###\s+4\.3\.3", line): return "4.3.3"
    if re.match(r"^#\s+5\s*-", line) or line == "# 5 - Storage": return "5"
    if re.match(r"^##\s+5\.1", line): return "5.1"
    if re.match(r"^###\s+5\.1\.1", line): return "5.1.1"
    if re.match(r"^###\s+5\.1\.2", line): return "5.1.2"
    if re.match(r"^###\s+5\.1\.3", line): return "5.1.3"
    if re.match(r"^###\s+5\.1\.4", line): return "5.1.4"
    if re.match(r"^##\s+5\.2", line): return "5.2"
    if re.match(r"^#\s+6\s*-", line) or line == "# 6 - Exploring optimal power mix design": return "6"
    if re.match(r"^##\s+6\.1", line): return "6.1"
    if re.match(r"^##\s+6\.2", line): return "6.2"
    if re.match(r"^###\s+6\.2\.1", line): return "6.2.1"
    if re.match(r"^###\s+6\.2\.2", line): return "6.2.2"
    if re.match(r"^###\s+6\.2\.3", line): return "6.2.3"
    if re.match(r"^###\s+6\.2\.4", line): return "6.2.4"
    if re.match(r"^##\s+6\.3", line): return "6.3"
    if re.match(r"^#\s+7\s*-", line) or "Choice a power mix" in line: return "7"
    if re.match(r"^##\s+7\.1", line): return "7.1"
    if re.match(r"^##\s+7\.3", line): return "7.3"
    if re.match(r"^##\s+7\.4", line): return "7.4"
    return None

def rewrite_heading_text(cell, new_heading):
    new_cell = dict(cell)
    source = list(new_cell.get("source", []))
    if source:
        source[0] = new_heading + "\n" if not new_heading.endswith("\n") else new_heading
    new_cell["source"] = source
    return new_cell

def main():
    with open("data_crunch.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    print(f"Read {len(cells)} cells.")
    
    # Classify all cells into sections
    sections = {}
    current_sec = "0"
    
    for cell in cells:
        source = cell.get("source", [])
        if cell["cell_type"] == "markdown" and source:
            first_line = source[0].strip()
            detected_sec = get_section_from_heading(first_line)
            if detected_sec is not None:
                current_sec = detected_sec
        
        if current_sec not in sections:
            sections[current_sec] = []
        sections[current_sec].append(cell)
        
    print("Classified sections:")
    for sec_id, sec_cells in sections.items():
        print(f"  {sec_id}: {len(sec_cells)} cells")
        
    # Build the final restructured cell list
    new_cells = []
    
    # 0. Notebook header intro / imports / study case choice
    if "0" in sections:
        new_cells.extend(sections["0"])
    if "1" in sections:
        # Update `# 1 - Imports` header
        import_header = rewrite_heading_text(sections["1"][0], "# Imports")
        new_cells.append(import_header)
        new_cells.extend(sections["1"][1:])
    if "2" in sections:
        # Update `# 2 - Choice of your study case` header
        case_header = rewrite_heading_text(sections["2"][0], "# Choice of your study case")
        new_cells.append(case_header)
        new_cells.extend(sections["2"][1:])
        
    # --- CHAPTER 1 ---
    new_cells.append(create_markdown_cell("chap1_title", ["# Chapter 1: Power demand"]))
    # Section 3.1 - Gross demand data
    if "3.1" in sections:
        hdr = rewrite_heading_text(sections["3.1"][0], "## 1.1 - Download and preparation of gross demand data")
        new_cells.append(hdr)
        new_cells.extend(sections["3.1"][1:])
    # Section 3.2 - Data Analysis (Average weather scenario)
    if "3.2" in sections:
        hdr = rewrite_heading_text(sections["3.2"][0], "## 1.2 - Data Analysis (Average weather scenario)")
        new_cells.append(hdr)
        new_cells.extend(sections["3.2"][1:])
    # Section 3.2.1 - Raw demand plots
    if "3.2.1" in sections:
        hdr = rewrite_heading_text(sections["3.2.1"][0], "### 1.2.1 - Raw demand plots")
        new_cells.append(hdr)
        new_cells.extend(sections["3.2.1"][1:])
        
    # Section 3.2.2 - Useful indicators for profile analysis (MODIFIED)
    # We will split seasonal indicators into Appendix A.1
    # First, let's process 3.2.2 cells for Chapter 1
    if "3.2.2" in sections:
        sec_322_cells = sections["3.2.2"]
        # In Chapter 1, we want indicators without seasonal variation in descriptions and plot
        # Let's clone and modify cell 1 (indicators list)
        cell_desc = dict(sec_322_cells[1])
        cell_desc["source"] = [
            "Let's compute basic statistics from load profile:\n",
            "- Annual Gross Demand\n",
            "- Peak Load\n",
            "- Daily Variation (maximal load spread in a single day)\n"
        ]
        
        # Cell 3 is the computation code block (d862a668). We keep it intact since it calculates results_df.
        cell_comp = sec_322_cells[3]
        
        # Cell 5 (Plotting the indicators) and cell 6 (The Plotly 2x2 grid). We replace the plot with 1x3 grid.
        cell_plot_desc = sec_322_cells[4]
        
        cell_plot_code = dict(sec_322_cells[5])
        cell_plot_code["source"] = [
            'fig = make_subplots(\n',
            '    rows=1, cols=3,\n',
            '    subplot_titles=[\n',
            '        "Total Annual Demand (TWh)",\n',
            '        "Peak Demand (GW)",\n',
            '        "Max Daily Variation (GW)"\n',
            '    ]\n',
            ')\n',
            '\n',
            'fig.add_trace(\n',
            '    go.Bar(\n',
            '        x=results_df.index,\n',
            '        y=results_df["Total_Demand_TWh"]/1e6,\n',
            '        text=(results_df["Total_Demand_TWh"]/1e6).round(0),\n',
            '        textposition="auto",\n',
            '        name="Total Demand (TWh)"\n',
            '    ),\n',
            '    row=1, col=1\n',
            ')\n',
            '\n',
            'fig.add_trace(\n',
            '    go.Bar(\n',
            '        x=results_df.index,\n',
            '        y=results_df["Peak_Load_GW"]/1e3,\n',
            '        text=(results_df["Peak_Load_GW"]/1e3).round(0),\n',
            '        textposition="auto",\n',
            '        name="Peak Load (GW)"\n',
            '    ),\n',
            '    row=1, col=2\n',
            ')\n',
            '\n',
            'fig.add_trace(\n',
            '    go.Bar(\n',
            '        x=results_df.index,\n',
            '        y=results_df["Max_Daily_Variation_GW"]/1e3,\n',
            '        text=(results_df["Max_Daily_Variation_GW"]/1e3).round(1),\n',
            '        textposition="auto",\n',
            '        name="Max Daily Variation (GW)"\n',
            '    ),\n',
            '    row=1, col=3\n',
            ')\n',
            '\n',
            'fig.update_layout(\n',
            '    title=f"Electricity Demand in {country_choice.capitalize()} by Year (Avg. Weather Scenario)",\n',
            '    template="plotly_white",\n',
            '    height=450,\n',
            '    width=1100,\n',
            '    showlegend=False\n',
            ')\n',
            '\n',
            'fig.update_xaxes(title_text="Year")\n',
            '\n',
            'fig.show()'
        ]
        
        # Assemble modified 3.2.2 cells for Chapter 1
        hdr = rewrite_heading_text(sec_322_cells[0], "### 1.2.2 - Useful indicators for profile analysis")
        new_cells.append(hdr)
        new_cells.append(cell_desc)
        new_cells.append(sec_322_cells[2]) # "Computing the indicators"
        new_cells.append(cell_comp)
        new_cells.append(cell_plot_desc)
        new_cells.append(cell_plot_code)
        
        # Add basic questions (cell 10 of 3.2.2 has questions: "Why are these indicators relevant?")
        questions_cell = sec_322_cells[10]
        new_cells.append(questions_cell)
        
    # Section 3.2.3 - Load Duration Curve
    if "3.2.3" in sections:
        hdr = rewrite_heading_text(sections["3.2.3"][0], "### 1.2.3 - Load Duration Curve")
        new_cells.append(hdr)
        new_cells.extend(sections["3.2.3"][1:])
        
    # Section 3.4 - Risk management and unsupplied energy (Moved up to Chapter 1)
    if "3.4" in sections:
        hdr = rewrite_heading_text(sections["3.4"][0], "## 1.3 - Risk management and unsupplied energy")
        new_cells.append(hdr)
        new_cells.extend(sections["3.4"][1:])
        
    # --- CHAPTER 2 ---
    new_cells.append(create_markdown_cell("chap2_title", ["# Chapter 2: Generation - Controllable generation technologies"]))
    if "4.1" in sections:
        hdr = rewrite_heading_text(sections["4.1"][0], "## 2.1 - Notion of controllable and variable generation")
        new_cells.append(hdr)
        new_cells.extend(sections["4.1"][1:])
    if "4.2.1" in sections:
        hdr = rewrite_heading_text(sections["4.2.1"][0], "## 2.2 - Controllable generation technologies (overview)")
        new_cells.append(hdr)
        new_cells.extend(sections["4.2.1"][1:])
        
    # --- CHAPTER 3 ---
    new_cells.append(create_markdown_cell("chap3_title", ["# Chapter 3: Boiteux stacking method (1/2) - only with controllable generation technologies"]))
    if "4.2.3" in sections:
        hdr = rewrite_heading_text(sections["4.2.3"][0], "## 3.1 - Sizing a 100% thermal/controllable system with Boiteux stacking")
        new_cells.append(hdr)
        new_cells.extend(sections["4.2.3"][1:])
        
    # --- CHAPTER 4 ---
    new_cells.append(create_markdown_cell("chap4_title", ["# Chapter 4: Generation - Renewable generation technologies"]))
    if "4.3.1" in sections:
        hdr = rewrite_heading_text(sections["4.3.1"][0], "## 4.1 - Variable Renewable Energy Sources (vRES) profiles")
        new_cells.append(hdr)
        new_cells.extend(sections["4.3.1"][1:])
        
    # --- CHAPTER 5 ---
    new_cells.append(create_markdown_cell("chap5_title", ["# Chapter 5: Boiteux stacking method (2/2) - with controllable and renewable generation technologies"]))
    if "4.3.2" in sections:
        hdr = rewrite_heading_text(sections["4.3.2"][0], "## 5.1 - Stacking under vRES: Impacts of net demand")
        new_cells.append(hdr)
        new_cells.extend(sections["4.3.2"][1:])
        
    # --- APPENDIX A ---
    new_cells.append(create_markdown_cell("appa_title", ["# Appendix A: Advanced Demand Analysis & Climatic Years"]))
    new_cells.append(create_markdown_cell("appa1_title", ["## A.1 - Seasonal and Advanced Demand Variations"]))
    
    # Let's add the seasonal variation indicators explanation and standalone seasonal variation plot
    seasonal_desc = create_markdown_cell("appa1_seasonal_desc", [
        "### Seasonal Demand Variation\n",
        "Let's analyze the seasonal demand variations by calculating the monthly average load spread (difference between the months with the highest and lowest average demand)."
    ])
    new_cells.append(seasonal_desc)
    
    # We create the standalone plot code for Seasonal variation
    seasonal_plot = {
        "cell_type": "code",
        "execution_count": None,
        "id": "seasonal_plot_standalone",
        "metadata": {},
        "outputs": [],
        "source": [
            'fig = go.Figure()\n',
            '\n',
            'fig.add_trace(\n',
            '    go.Bar(\n',
            '        x=results_df.index,\n',
            '        y=results_df["Seasonal_Variation_GW"]/1e3,\n',
            '        text=(results_df["Seasonal_Variation_GW"]/1e3).round(1),\n',
            '        textposition="auto",\n',
            '        name="Seasonal Variation (GW)",\n',
            '        marker_color="indianred"\n',
            '    )\n',
            ')\n',
            '\n',
            'fig.update_layout(\n',
            '    title=f"Seasonal Demand Variation in {country_choice.capitalize()} by Year (Avg. Weather Scenario)",\n',
            '    yaxis_title="Seasonal Variation (GW)",\n',
            '    template="plotly_white",\n',
            '    height=450,\n',
            '    width=600,\n',
            '    showlegend=False\n',
            ')\n',
            '\n',
            'fig.update_xaxes(title_text="Year")\n',
            '\n',
            'fig.show()'
        ]
    }
    new_cells.append(seasonal_plot)
    
    # Now let's move the daily/weekly variation boxplots from 3.2.2 (cells 7, 8, 9)
    if "3.2.2" in sections:
        sec_322_cells = sections["3.2.2"]
        # Cells 6, 7, 8, 9 are: Optional description, optional indicators markdown, code cell boxplots, and questions
        new_cells.append(sec_322_cells[6]) # Optional what other indicators
        new_cells.append(sec_322_cells[7]) # Optional indicators markdown
        new_cells.append(sec_322_cells[8]) # Code boxplots
        new_cells.append(sec_322_cells[9]) # Questions
        
    # Climatic years concept section (A.2)
    new_cells.append(create_markdown_cell("appa2_title", ["## A.2 - Demand Analysis including Climatic Years (Inter-annual dispersion)"]))
    if "3.3" in sections:
        hdr = rewrite_heading_text(sections["3.3"][0], "### A.2.1 - Data Analysis including Climatic Years")
        new_cells.append(hdr)
        new_cells.extend(sections["3.3"][1:])
    if "3.3.1" in sections:
        hdr = rewrite_heading_text(sections["3.3.1"][0], "### A.2.2 - Raw demand plots across climatic years")
        new_cells.append(hdr)
        new_cells.extend(sections["3.3.1"][1:])
    if "3.3.2" in sections:
        hdr = rewrite_heading_text(sections["3.3.2"][0], "### A.2.3 - Useful indicators for profile analysis across climatic years")
        new_cells.append(hdr)
        new_cells.extend(sections["3.3.2"][1:])
    if "3.3.3" in sections:
        hdr = rewrite_heading_text(sections["3.3.3"][0], "### A.2.4 - Load duration curve across climatic years")
        new_cells.append(hdr)
        new_cells.extend(sections["3.3.3"][1:])
        
    # --- APPENDIX B ---
    new_cells.append(create_markdown_cell("appb_title", ["# Appendix B: Controllable technologies - Detailed economic and environmental characteristics"]))
    if "4.2.2" in sections:
        hdr = rewrite_heading_text(sections["4.2.2"][0], "## B.1 - Economic and environmental characteristics of controllable technologies")
        new_cells.append(hdr)
        new_cells.extend(sections["4.2.2"][1:])
        
    # --- APPENDIX C ---
    new_cells.append(create_markdown_cell("appc_title", ["# Appendix C: Renewable technologies - Detailed economic and environmental characteristics"]))
    if "4.3.3" in sections:
        hdr = rewrite_heading_text(sections["4.3.3"][0], "## C.1 - Economic and environmental characteristics of renewable technologies")
        new_cells.append(hdr)
        new_cells.extend(sections["4.3.3"][1:])
        
    # --- APPENDIX D ---
    new_cells.append(create_markdown_cell("appd_title", ["# Appendix D: Storage"]))
    if "5.1" in sections:
        hdr = rewrite_heading_text(sections["5.1"][0], "## D.1 - Batteries")
        new_cells.append(hdr)
        new_cells.extend(sections["5.1"][1:])
    if "5.1.1" in sections:
        hdr = rewrite_heading_text(sections["5.1.1"][0], "### D.1.1 - Impact without vRES")
        new_cells.append(hdr)
        new_cells.extend(sections["5.1.1"][1:])
    if "5.1.2" in sections:
        hdr = rewrite_heading_text(sections["5.1.2"][0], "### D.1.2 - Impact with vRES")
        new_cells.append(hdr)
        new_cells.extend(sections["5.1.2"][1:])
    if "5.1.3" in sections:
        hdr = rewrite_heading_text(sections["5.1.3"][0], "### D.1.3 - Economic parameters")
        new_cells.append(hdr)
        new_cells.extend(sections["5.1.3"][1:])
    if "5.1.4" in sections:
        hdr = rewrite_heading_text(sections["5.1.4"][0], "### D.1.4 - Environmental considerations")
        new_cells.append(hdr)
        new_cells.extend(sections["5.1.4"][1:])
    if "5.2" in sections:
        hdr = rewrite_heading_text(sections["5.2"][0], "## D.2 - Pumped Hydro")
        new_cells.append(hdr)
        new_cells.extend(sections["5.2"][1:])
        
    # --- APPENDIX E ---
    new_cells.append(create_markdown_cell("appe_title", ["# Appendix E: Optimization and choice of power mix"]))
    if "6.1" in sections:
        hdr = rewrite_heading_text(sections["6.1"][0], "## E.1 - Optimal power mix with vRES deployment target")
        new_cells.append(hdr)
        new_cells.extend(sections["6.1"][1:])
    if "6.2" in sections:
        hdr = rewrite_heading_text(sections["6.2"][0], "## E.2 - Optimal power mix (without vRES target)")
        new_cells.append(hdr)
        new_cells.extend(sections["6.2"][1:])
    if "6.2.1" in sections:
        hdr = rewrite_heading_text(sections["6.2.1"][0], "### E.2.1 - First approach without additional constraint")
        new_cells.append(hdr)
        new_cells.extend(sections["6.2.1"][1:])
    if "6.2.2" in sections:
        hdr = rewrite_heading_text(sections["6.2.2"][0], "### E.2.2 - Constraints on emission reduction")
        new_cells.append(hdr)
        new_cells.extend(sections["6.2.2"][1:])
    if "6.2.3" in sections:
        hdr = rewrite_heading_text(sections["6.2.3"][0], "### E.2.3 - Political choices")
        new_cells.append(hdr)
        new_cells.extend(sections["6.2.3"][1:])
    if "6.2.4" in sections:
        hdr = rewrite_heading_text(sections["6.2.4"][0], "### E.2.4 - *(Optional) Additional (LP) constraints on nuclear*")
        new_cells.append(hdr)
        new_cells.extend(sections["6.2.4"][1:])
    if "6.3" in sections:
        hdr = rewrite_heading_text(sections["6.3"][0], "## E.3 - Choice of a mix considering various climatic years")
        new_cells.append(hdr)
        new_cells.extend(sections["6.3"][1:])
    if "7.1" in sections:
        hdr = rewrite_heading_text(sections["7.1"][0], "## E.4 - Comparison with ERAA power mix")
        new_cells.append(hdr)
        new_cells.extend(sections["7.1"][1:])
    if "7.3" in sections:
        hdr = rewrite_heading_text(sections["7.3"][0], "## E.5 - Capacity Expansion : start from current power mix")
        new_cells.append(hdr)
        new_cells.extend(sections["7.3"][1:])
    if "7.4" in sections:
        hdr = rewrite_heading_text(sections["7.4"][0], "## E.6 - Choice of power mix for PyPSA")
        new_cells.append(hdr)
        new_cells.extend(sections["7.4"][1:])
        
    nb["cells"] = new_cells
    
    # Save the updated notebook
    with open("data_crunch.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Restructured notebook saved successfully with {len(new_cells)} cells!")

if __name__ == "__main__":
    main()
