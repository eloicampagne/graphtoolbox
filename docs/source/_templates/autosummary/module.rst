{{ fullname | escape | underline }}

{% if modules %}
.. automodule:: {{ fullname }}

.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% else %}
.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
{% endif %}
