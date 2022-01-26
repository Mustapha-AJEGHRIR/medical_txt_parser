def search_query(query, filters={}, top=10):
    return [
        {
            "filename": f"018636330_DH_{i}.txt",
            "preview": """DISCHARGE MEDICATIONS :
She was discharged on the following medications .
1. Colace , 100 mg PO bid .
2. Zantac , 150 mg PO bid .""",
            "metadata": {
                "age": 30,
                "sexe": "F",
                "birthdate": "01/01/1980",
                "admission_date": "01/01/2020",
                "discharge_date": "01/01/2021",
            },
        } for i in range(100)
    ]
