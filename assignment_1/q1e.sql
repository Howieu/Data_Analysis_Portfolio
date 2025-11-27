-- Emma Round 1

-- === ROLL 1: 6 ===
UPDATE players
SET credits = credits + CASE WHEN (
    current_location_id + 6
) >= 20 THEN 100 ELSE 0 END,
    current_location_id = (
        current_location_id + 6
    ) % 20
WHERE name = 'Emma';

UPDATE players 
SET current_location_id = 7, is_suspended = 1 
WHERE name = 'Emma' AND current_location_id = 17;

INSERT INTO audit_log (round_num, player_id, location_id, balance_after_turn, action_type)
SELECT 1, player_id, current_location_id, credits, 'ROLL_6_NO_EFFECT' 
FROM players 
WHERE name = 'Emma';

-- === ROLL 2: 4 ===
UPDATE players
SET credits = credits + CASE WHEN (
    current_location_id + 4
) >= 20 THEN 100 ELSE 0 END,
    current_location_id = (
        current_location_id + 4
    ) % 20
WHERE name = 'Emma';

UPDATE players 
SET current_location_id = 7, is_suspended = 1 
WHERE name = 'Emma' AND current_location_id = 17;

-- Try Buy
UPDATE players 
SET credits = credits - (
    SELECT cost FROM buildings WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Emma')
)
WHERE name = 'Emma' AND EXISTS (
    SELECT 1 FROM buildings WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Emma') AND owner_id IS NULL
);

UPDATE buildings 
SET owner_id = (SELECT player_id FROM players WHERE name = 'Emma') 
WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Emma') AND owner_id IS NULL;

-- Try Pay Rent
UPDATE players 
SET credits = credits - COALESCE((
    SELECT CASE 
        WHEN (
            SELECT COUNT(*) FROM buildings b2 WHERE b2.group_id = b1.group_id AND b2.owner_id = b1.owner_id
        ) = (
            SELECT COUNT(*) FROM buildings b3 WHERE b3.group_id = b1.group_id
        ) 
        THEN b1.tuition_fee * 2 ELSE b1.tuition_fee END
    FROM buildings b1
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Emma')
      AND owner_id IS NOT NULL AND owner_id != (SELECT player_id FROM players WHERE name = 'Emma')
), 0)
WHERE name = 'Emma';

-- Owner Receives Rent
UPDATE players 
SET credits = credits + COALESCE((
    SELECT CASE 
        WHEN (
            SELECT COUNT(*) FROM buildings b2 WHERE b2.group_id = b1.group_id AND b2.owner_id = b1.owner_id
        ) = (
            SELECT COUNT(*) FROM buildings b3 WHERE b3.group_id = b1.group_id
        ) 
        THEN b1.tuition_fee * 2 ELSE b1.tuition_fee END
    FROM buildings b1
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Emma')
), 0)
WHERE player_id = (
    SELECT owner_id 
    FROM buildings 
    WHERE location_id = (SELECT current_location_id FROM players WHERE name = 'Emma') 
    AND owner_id IS NOT NULL 
    AND owner_id != (SELECT player_id FROM players WHERE name = 'Emma')
);

-- Special Events
UPDATE players SET credits = credits + CASE 
    WHEN current_location_id = 3 THEN -30 
    WHEN current_location_id = 16 THEN -100 
    WHEN current_location_id = 6 THEN 75 
    ELSE 0 
END 
WHERE name = 'Emma' AND current_location_id IN (3, 16, 6);

INSERT INTO audit_log (round_num, player_id, location_id, balance_after_turn, action_type)
SELECT 1, player_id, current_location_id, credits, 'TURN_END' 
FROM players WHERE name = 'Emma';