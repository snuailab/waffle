from abc import abstractmethod

import streamlit as st
from src.schema import PageInfo


class BasePage:
    def __init__(
        self,
        title: str,
        subtitle: str = None,
        description: str = None,
    ):
        self.title = title
        self.subtitle = subtitle
        self.description = description
    
    def render_title(self):
        if self.title:
            return st.title(self.title)
    
    def render_sub_title(self):
        if self.subtitle:
            return st.subheader(self.subtitle)

    def render_description(self):
        if self.description:
            return st.text(self.description)

    @abstractmethod
    def render_content(self):
        raise NotImplementedError

    def render(self):
        self.render_title()
        self.render_sub_title()
        self.render_description()
        self.render_content()

    def __call__(self):
        self.render()
