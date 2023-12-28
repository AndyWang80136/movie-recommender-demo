WITH cte AS (
    SELECT 
        items.*,
        {{ user_id }} AS user_id
    FROM items
    WHERE items.item_id IN {{ item_ids }}
)

SELECT 
    *,
    occupation_id AS occupation,
    age_id AS age,
    (SELECT MAX(rating_timestamp) FROM ratings WHERE user_id = {{ user_id }}) AS timestamp
FROM cte
JOIN users USING (user_id)