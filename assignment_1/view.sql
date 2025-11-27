DROP VIEW IF EXISTS leaderboard;

CREATE VIEW leaderboard AS
SELECT
    p.name,
    LOWER(REPLACE(REPLACE(l.name, '''', ''), ' ', '_')) AS location,
    p.credits,
    COALESCE(b_list.owned_buildings, '') AS buildings

FROM
    players p
JOIN
    locations l ON p.current_location_id = l.location_id

LEFT JOIN (
    SELECT
        owner_id,
        GROUP_CONCAT(clean_name, ', ') AS owned_buildings
    FROM (
        SELECT
            b.owner_id,
            LOWER(REPLACE(REPLACE(loc.name, '''', ''), ' ', '_')) AS clean_name
        FROM
            buildings b
        JOIN
            locations loc ON b.location_id = loc.location_id
        ORDER BY
            b.location_id ASC
    )
    GROUP BY
        owner_id
) b_list ON p.player_id = b_list.owner_id

ORDER BY
    p.credits DESC;

SELECT * FROM leaderboard;