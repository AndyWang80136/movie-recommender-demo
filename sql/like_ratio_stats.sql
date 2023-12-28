SELECT 
    likes_table.item_id,
    likes_table.attribute_name,
    likes_table.attribute_value,
    ROUND(num_likes::NUMERIC/NULLIF(num_ratings, 0), 2) AS like_ratio
FROM num_likes_user_domain likes_table
JOIN num_ratings_user_domain ratings_table ON 
    ratings_table.attribute_name = likes_table.attribute_name AND
    ratings_table.attribute_value = likes_table.attribute_value AND 
    ratings_table.item_id = likes_table.item_id
WHERE (likes_table.attribute_name = 'gender' AND likes_table.attribute_value = '{{ gender }}') OR 
    (likes_table.attribute_name = 'age_interval' AND likes_table.attribute_value = '{{ age_interval }}') OR
    (likes_table.attribute_name = 'occupation' AND likes_table.attribute_value = '{{ occupation }}')
ORDER BY item_id ASC
