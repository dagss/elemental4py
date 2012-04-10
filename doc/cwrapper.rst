About the C wrapper around Elemental
====================================


Choice of templating language
-----------------------------

C++ templates are not suitable when what one wants to do is
*instantiate* the C++ templates in the first place. Code needs to be
generated in the cpp file, and this is something C++ templates can't
do.

Jinja2 was chosen as the templating language, after first trying
Tempita and Mako. The problem with Tempita is that it does not
support passing "scope bodies" as callback arguments to functions.
Mako was more powerful than Jinja2, but the syntax was simply too
terrible, and Jinja2 is as powerful when combined with an external
Python module.
