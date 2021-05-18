def fetch_style(name):
    if name == "above-values":
        return {
                                "text-align": "center",
                                "background-color": "#333333",
                                "border-radius": "15px",
                                "border": "2px solid #33333",
                                "padding": "5px",
                            }
    elif name == "dropdowns":
        return {
            "width": "100 %",
            "background-color": "#E1E1E1",
            "font-size": "75%",
            "color": "#000000"
        }
    elif name == "actual-values":
        return {"text-align": "center", "color": "#316AC3"}

