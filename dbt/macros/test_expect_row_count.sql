{# Asserts a model has exactly the expected number of rows. Guards against a
   join or filter silently dropping customers between raw and features. #}
{% test expect_row_count(model, expected) %}

select count(*) as actual_rows
from {{ model }}
having count(*) != {{ expected }}

{% endtest %}
