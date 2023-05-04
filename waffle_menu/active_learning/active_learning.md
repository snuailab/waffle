# Active Learning Base Class
all active learning methods are inherited from `ActiveLearning` class. Only initialize methods are different. It means that you can use any active learning method by calling sample method of `ActiveLearning` class.

::: waffle_menu.active_learning.ActiveLearning
    handler: python
    options:
        members:
            - sample
        show_source: false

# RandomSampling
::: waffle_menu.active_learning.RandomSampling
    handler: python
    options:
        members:
            - __init__
        show_source: false

# EntropySampling
::: waffle_menu.active_learning.EntropySampling
    handler: python
    options:
        members:
            - __init__
        show_source: false

# PL2NSampling
::: waffle_menu.active_learning.PL2NSampling
    handler: python
    options:
        members:
            - __init__
        show_source: false