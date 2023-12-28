SELECT 
    user_id,
    item_id,
    rating,
    CASE WHEN rating > 3 THEN 1 ELSE 0 END AS label,
    prob,
    movie_title,
    movie_genres,
    age_id AS age,
    occupation_id AS occupation,
    gender,
    age_interval AS age_display,
    occupations.occupation AS occupation_display,
    gender AS gender_display
FROM test_results
JOIN ratings USING (rating_id)
JOIN users USING (user_id)
JOIN items USING (item_id)
JOIN occupations USING (occupation_id)
JOIN ages USING (age_id)