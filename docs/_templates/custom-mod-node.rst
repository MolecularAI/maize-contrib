{{ name | escape | underline }}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Nodes') }}

   .. autosummary::
      :toctree: {{ fullname }}
      :template: custom-node.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
