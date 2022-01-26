import streamlit as st 
from inference import detect_concept 


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

def main():
	"""Concept Streamlit App"""

	st.title("Concept Checker")

	activities = ["Concept","Assertion","Relation"]
	choice = st.sidebar.selectbox("Select Activity", activities)

	if choice == 'Concept':
        output = []
		st.subheader("Medical Concepts")
		raw_text = st.text_area("Enter Text Here","Type Here")
		if st.button("Summarize"):
            ner_result = detect_concept(raw_text)
            # TODO: format the result for NER
            result = "..."
			st.write(result)

	if choice == 'Assertion':


	if choice == 'Relation':
				
		


if __name__ == '__main__':
	main()