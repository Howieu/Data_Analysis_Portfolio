-- Stewart Round 3

-- 1. Roll 6 (Move)
UPDATE players
SET credits = credits + CASE WHEN (current_location_id + 6) >= 20 THEN 100 ELSE 0 END,
    current_location_id = (current_location_id + 6) % 20
WHERE name = 'Stewart';

-- 2. JAIL TRIGGER
-- Stewart lands on 17 -> Sent to 7 -> Suspended -> Turn Ends
UPDATE players 
SET current_location_id = 7, is_suspended = 1   
WHERE name = 'Stewart' AND current_location_id = 17;

-- 3. Stop! No second roll allowed.
INSERT INTO audit_log (round_num, player_id, location_id, balance_after_turn, action_type)
SELECT 3, player_id, current_location_id, credits, 'JAILED_TURN_END' 
FROM players WHERE name = 'Stewart';