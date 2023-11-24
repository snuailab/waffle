import streamlit as st


def generate_component(name: str, type_: type, default=None, key=None):

    if type_ == bool:
        return st.checkbox(name, value=default, key=key)

    if type_ == str:
        return st.text_input(name, value=default, key=key)

    if type_ == int:
        return st.number_input(name, value=default, key=key)

    if type_ == float:
        return st.number_input(name, value=default, key=key)

    # check if type_ is list something
    if hasattr(type_, "__origin__") and type_.__origin__ == list:
        return st.multiselect(name, options=default, key=key)
