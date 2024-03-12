SELECT 
    users.user_id,
    users.gender,
    ages.age_interval,
    occupations.occupation
FROM users
JOIN occupations USING (occupation_id)
JOIN ages USING (age_id)
