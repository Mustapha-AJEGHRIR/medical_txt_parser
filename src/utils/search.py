import random

def search_query(query, filters={}, top=10):

    results = [
        {
            "filename": f"018636330_DH_{i}.txt",
            "preview": """DISCHARGE MEDICATIONS :
She was discharged on the following medications .
1. Colace , 100 mg PO bid .
2. Zantac , 150 mg PO bid .""",
            "metadata": {
                "age": 30,
                "sexe": f"{random.choice(['F', 'M'])}",
                "birthdate": f"1{900+i}-01-01",
                "admission_date": f"1{900+i}-01-01",
                "discharge_date": f"1{900+i}-01-01",
            },
        } for i in range(150)
    ]
    # filter results
    range_filters = ["age", "birthdate", "admission_date", "discharge_date"]
    multiselect_filters = ["sexe"]
    filtered_results = []
    for result in results:
        valid = True
        for key in range_filters:
            if key in filters:
                if filters[key][0] > result["metadata"][key] or filters[key][1] < result["metadata"][key]:
                    valid = False
                    break
        if valid:
            for key in multiselect_filters:
                if key in filters:
                    if result["metadata"][key] not in filters[key]:
                        valid = False
                        break
        if valid:
            filtered_results.append(result)

    count_filtered = len(filtered_results)
    filtered_results = filtered_results[:top]
    return filtered_results, count_filtered