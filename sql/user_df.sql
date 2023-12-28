SELECT 
    users.user_id,
    users.gender,
    ages.age_interval,
    occupations.occupation,
    train_count,
    val_count,
    test_count
FROM user_phase_count
JOIN users USING (user_id)
JOIN occupations USING (occupation_id)
JOIN ages USING (age_id)
WHERE test_count != 0
ORDER BY train_count DESC
