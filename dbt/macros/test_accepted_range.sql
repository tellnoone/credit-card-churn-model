{# Self-contained numeric range test. Avoids a dbt_utils dependency (and the
   network fetch that `dbt deps` would need) for one small check. #}
{% test accepted_range(model, column_name, min_value, max_value) %}

select *
from {{ model }}
where {{ column_name }} is null
   or {{ column_name }} < {{ min_value }}
   or {{ column_name }} > {{ max_value }}

{% endtest %}
