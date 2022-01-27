""" Search engine UI. 
based on https://betterprogramming.pub/build-a-search-engine-for-medium-stories-using-streamlit-and-elasticsearch-b6e717819448
"""
import os
import re
import json
from dotenv import load_dotenv
load_dotenv()
import datetime


import itertools
import requests
from PIL import Image
import base64
import streamlit as st
from st_utils import visualize_record

# https://gist.github.com/treuille/2ce0acb6697f205e44e3e0f576e810b7
def paginator(label, articles, articles_per_page=10, on_sidebar=True):
    """Lets the user paginate a set of article.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    article : Iterator[Any]
        The articles to display in the paginator.
    articles_per_page: int
        The number of articles to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.

    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the article on that page*, including
        the item's index.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    articles = list(articles)
    n_pages = (len(articles) - 1) // articles_per_page + 1
    page_format_func = lambda i: f"Results {i*10} to {i*10 +10 -1}"
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the articles in the page to let the user display them.
    min_index = page_number * articles_per_page
    max_index = min_index + articles_per_page

    return itertools.islice(enumerate(articles), min_index, max_index)


if "selected_record" not in st.session_state:
    st.session_state["selected_record"] = None


def set_record(record):
    st.session_state["selected_record"] = record



if not st.session_state["selected_record"]: # search engine page
    st.set_page_config(
        page_title="Records Database",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="auto",
    )
    ### SIDEBAR
    st.sidebar.markdown('# Filters')
    
    age_range = st.sidebar.slider('Age', min_value=0, max_value=100, value=(0,100))
    sexe = st.sidebar.multiselect('Sexe', ['F', 'M', 'N/A'], default=['F', 'M', 'N/A'])
    birthdate = st.sidebar.date_input('Birthdate', value=[datetime.date(1900, 1, 1), datetime.date(2021, 1, 1)])
    admission_date = st.sidebar.date_input('Admission date', value=[datetime.date(1900, 1, 1), datetime.date(2021, 1, 1)])
    discharge_date = st.sidebar.date_input('Discharge date', value=[datetime.date(1900, 1, 1), datetime.date(2021, 1, 1)])

    # clear filters
    # if st.sidebar.button('Clear filters'):
    #     st.session_state["selected_record"] = None
    #     st.sidebar.success('Filters cleared')

    st.markdown(
        "<h1 style='text-align: center; '>Patients records database</h1>",
        unsafe_allow_html=True,
    )
    # st.markdown("<h2 style='text-align: center; '>Stay safe</h2>", unsafe_allow_html=True)

    # Logo
    # logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facemask.jpg")
    # robeco_logo = Image.open(logo_path)
    # st.image(robeco_logo, use_column_width=True)

    # Search bar
    search_query = st.text_input("Search for a patient's record", value="", max_chars=None, key=None, type="default")


    # Search API
    index_name = "train-index"
    endpoint = os.environ["ENDPOINT"]
    headers = {
        "Content-Type": "application/json",
        "api-key": "password",
    }
    search_url = f"{endpoint}/indexes/{index_name}/docs/search"
    filters = {
        "age": age_range,
        "sexe": sexe,
        "birthdate": birthdate,
        "admission_date": admission_date,
        "discharge_date": discharge_date,
    }
    search_body = {
        "query": search_query,
        "filters": json.dumps(filters, default=str),
        "top": 30,
    }

    if search_query != "":
        response = requests.post(search_url, headers=headers, json=search_body).json()

        record_list = []
        _ = [
            record_list.append(
                {"filename": record["filename"], "preview": record["preview"], "metadata": record["metadata"], "id": record["id"]}
            )
            for record in response.get("value")
        ]

        # filter results

        if record_list:
            st.write(f'Search results ({response.get("count")}):')

            if response.get("count") > 100:
                shown_results = 100
            else:
                shown_results = response.get("count")

            for i, record in paginator(
                f"Select results (showing {shown_results} of {response.get('count')} results)",
                record_list,
            ):
                st.button(f"View record {record['filename']}", on_click=lambda: set_record(record), key=record["id"])
                st.write("**Filename:** %s" % (record["filename"]))
                st.write("**Preview:** %s" % (record["preview"]))
                st.markdown("---")
        else:
            st.write(f"No Search results, please try again with different keywords")

else:  # a record has been selected
    st.set_page_config(
        page_title=f"Record {st.session_state['selected_record']['filename']}",
        page_icon="👨‍⚕️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.button("Back", on_click=lambda: set_record(None))

    st.markdown(
        f"<h1 style='text-align: center; '>Patient record: {st.session_state['selected_record']['filename']}</h1>",
        unsafe_allow_html=True,
    )
    # select task
    task = st.selectbox("Task", ["concept", "assertion"], format_func=lambda x: {"concept": "Concepts detection", "assertion": "Assertions classification"}[x])
    visualize_record(st.session_state["selected_record"], task=task)
    
    # st.write(st.session_state["selected_record"])