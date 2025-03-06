-- Set package paths
package.path = package.path .. ";C:/Users/autodidactic/AppData/Roaming/luarocks/share/lua/5.1/?.lua"
package.cpath = package.cpath .. ";C:/Users/autodidactic/AppData/Roaming/luarocks/lib/lua/5.1/?.dll"

local socket = require("socket")

-- Define global variables
local MySocket = nil
local IPAddress = "127.0.0.1"
local Port = 5005  -- Ensure your listener is using this port


weapons_data = {
    ["{AB_250_2_SD_2}"] = {name = "AB 250-2 - 144 x SD-2, 250kg CBU with HE submunitions", weight = 280},
    ["{AB_250_2_SD_10A}"] = {name = "AB 250-2 - 17 x SD-10A, 250kg CBU with 10kg Frag/HE submunitions", weight = 220},
    ["{AB_500_1_SD_10A}"] = {name = "AB 500-1 - 34 x SD-10A, 500kg CBU with 10kg Frag/HE submunitions", weight = 470},
    ["{682A481F-0CB5-4693-A382-D00DD4A156D7}"] = {name = "R-60 (AIM-9 Equivalent IR Missile)", weight = 43},
    ["{FC56DF80-9B09-44C5-8976-DCFAFF219062}"] = {name = "S-8 Rocket Pod", weight = 120},
    ["{0180F983-C14A-11d8-9897-000476191836}"] = {name = "FAB-100 Bomb", weight = 100},
    -- Add more weapons here
}



local function printTable(tbl, indent)
    indent = indent or 0
    if type(tbl) ~= "table" then
        log.write("Export.lua", log.INFO, string.rep(" ", indent) .. tostring(tbl))
        return
    end
    for k, v in pairs(tbl) do
        if type(v) == "table" then
            log.write("Export.lua", log.INFO, string.rep(" ", indent) .. tostring(k) .. " => {")
            printTable(v, indent + 4) -- Recursive call for nested tables
            log.write("Export.lua", log.INFO, string.rep(" ", indent) .. "}")
        else
            log.write("Export.lua", log.INFO, string.rep(" ", indent) .. tostring(k) .. " = " .. tostring(v))
        end
    end
end

function printtable2(tbl, indent)
    indent = indent or 0
    if type(tbl) ~= "table" then
        log.write("Export.lua", log.INFO, string.rep("  ", indent) .. tostring(tbl))
        return
    end
    for k, v in pairs(tbl) do
        if type(v) == "table" then
            log.write("Export.lua", log.INFO, string.rep("  ", indent) .. k .. " => {")
            printtable2(v, indent + 1)
            log.write("Export.lua", log.INFO, string.rep("  ", indent) .. "}")
        else
            log.write("Export.lua", log.INFO, string.rep("  ", indent) .. k .. " = " .. tostring(v))
        end
    end
end


local function getDistanceHaversine(lat1, lon1, lat2, lon2)
    local R = 6371 -- Earth's radius in KM
    local dLat = math.rad(lat2 - lat1)
    local dLon = math.rad(lon2 - lon1)

    local a = math.sin(dLat/2) * math.sin(dLat/2) +
              math.cos(math.rad(lat1)) * math.cos(math.rad(lat2)) *
              math.sin(dLon/2) * math.sin(dLon/2)

    local c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c -- Distance in KM
end


function LuaExportStart()
    log.write("Export.lua", log.INFO, "✅ Starting Lua Export")

    -- Use UDP socket (TCP not needed)
    MySocket = socket.udp()
    MySocket:setpeername(IPAddress, Port)  -- Remove the setoption line

    log.write("Export.lua", log.INFO, "✅ Connected to " .. IPAddress .. ":" .. Port)
end

function LuaExportBeforeNextFrame()
end

function LuaExportAfterNextFrame()
    if not MySocket then return end

    local IAS = LoGetIndicatedAirSpeed() or 0
    local RALT = LoGetAltitudeAboveGroundLevel() or 0
    local AoA = LoGetAngleOfAttack() or 0
    local altBar = LoGetAltitudeAboveSeaLevel() or 0
    local pitch, bank, yaw = LoGetADIPitchBankYaw()
    pitch = pitch or 0
    bank = bank or 0
    yaw = yaw or 0

    local aircraft = LoGetSelfData()

--         print("Aircraft Name: ", aircraft.Name)
--         print(": ", aircraft.LatLongAlt)
--         print(": ", aircraft.Vx, aircraft.Vy, aircraft.Vz)
--         print("Heading: ", aircraft.Heading)



--     local tws = LoGetTWSInfo() or 0
--     local target_info = LoGetTargetInformation() or 0
--     local locked_target_info = LoGetLockedTargetInformation() or 0
--     local f15_tws = LoGetF15_TWS_Contacts() or 0
--     local sighting = LoGetSightingSystemInfo() or 0
--     local wing_targets = LoGetWingTargets() or 0
    log.write("Export.lua", log.INFO, "✅ Sending Airspeed: " .. IAS)

--     MySocket:send(string.format("IAS: %.4f  RALT: %.4f AoA: %.4f altbar: %.4f pitch: %.4f bank: %.4f yaw: %.4f tws: %.4f target info: %.4f locked: %.4f f15: %.4f sighting: %.4f wing: %.4f\n", IAS, RALT, AoA, altBar,pitch, bank,yaw, tws, target_info, locked_target_info, f15_tws, sighting, wing_targets))
    MySocket:send(string.format("IAS: %.4f  RALT: %.4f AoA: %.4f altbar: %.4f pitch: %.4f bank: %.4f yaw: %.4f\n", IAS, RALT, AoA, altBar,pitch, bank,yaw))
--     if aircraft then
--         log.write("Export.lua", log.INFO, "📌 Aircraft Name: " .. tostring(aircraft.Name))
--         log.write("Export.lua", log.INFO, "📌 Position Lat: " .. tostring(aircraft.LatLongAlt.x))
-- --         log.write("Export.lua", log.INFO, tostring("📌 Position: Lat: ", aircraft.LatLongAlt.x))
--         log.write("Export.lua", log.INFO, "📌 Heading: " .. tostring(aircraft.Heading))
--
-- --         MySocket:send(string.format("Aircraft Name: %s  Position: %.4f Heading: %.4f \n", aircraft.Name, aircraft.LatLongAlt, aircraft.Heading))
--     end
    -- Get all world objects in the mission
    local worldObjects = LoGetWorldObjects()

--     -- Iterate through the objects
--     for objectID, object in pairs(worldObjects) do
--         -- Ensure the object has a valid name and position
--         if object.Name and object.LatLongAlt then
--             local latitude = tonumber(object.LatLongAlt.latitude) or 0
--             local longitude = tonumber(object.LatLongAlt.longitude) or 0
--             local altitude = tonumber(object.LatLongAlt.altitude) or 0
--             local heading = tonumber(object.Heading) or 0
--             local coalition = object.Coalition or "Unknown"
--
--             -- Log object info (for debugging)
--             log.write("Export.lua", log.INFO, string.format("📌 Object: %s | Coalition: %s | Lat: %.6f, Long: %.6f, Alt: %.2f, Heading: %.2f", object.Name, coalition, latitude, longitude, altitude, heading))
--
--             -- Send object information via socket
-- --             MySocket:send(string.format("Object: %s | Coalition: %s | Position: Lat: %.6f, Long: %.6f, Alt: %.2f | Heading: %.2f\n", object.Name, coalition, latitude, longitude, altitude, heading))
--         end
--     end


--     for objectID, object in pairs(worldObjects) do
--         log.write("Export.lua", log.INFO, "📌 DEBUG OBJECT STRUCTURE:")
--         printTable(object)
--         break -- Only print one object for now (to prevent log spam)
--     end
--
--     local worldObjects = LoGetWorldObjects()

    for objectID, object in pairs(worldObjects) do
        if object.Coalition == "Enemies" then
            my_enemy_id = objectID
            my_enemy_name = object.Name
            log.write("Export.lua", log.INFO, "🚨 Enemy Detected: " .. object.Name)
            log.write("Export.lua", log.INFO, string.format("📍 Position: Lat: %.6f, Long: %.6f, Alt: %.2f",
                object.LatLongAlt.Lat,
                object.LatLongAlt.Long,
                object.LatLongAlt.Alt
            ))
            log.write("Export.lua", log.INFO, "📏 Heading: " .. object.Heading)
        end
    end


        -- Example Usage:
    local myAircraft = LoGetSelfData()
    local enemyAircraft = LoGetWorldObjects()[my_enemy_id] -- Example ID of an enemy

    if myAircraft and enemyAircraft then
        local myPos = myAircraft.LatLongAlt
        local enemyPos = enemyAircraft.LatLongAlt

        local distanceKM = getDistanceHaversine(myPos.Lat, myPos.Long, enemyPos.Lat, enemyPos.Long)
        log.write("Export.lua", log.INFO, string.format("🌍 Haversine Distance to Enemy %s: %.2f km",my_enemy_name, distanceKM))
    end


--     local payload = LoGetPayloadInfo()  -- Get aircraft's payload info

    log.write("Export.lua", log.INFO, "🔹 Munition Info Retrieved!")

--     for i, weapon in ipairs(payload.Stations) do
--         if weapon.weapon then
--             log.write("Export.lua", log.INFO, string.format(
--                 "🚀 Station %d: %s | Count: %d",
--                 i, weapon.weapon.name, weapon.weapon.count))
--         else
--             log.write("Export.lua", log.INFO, string.format("📌 Station %d: Empty", i))
--         end
--     end

--     local payload = LoGetPayloadInfo()
--     log.write("Export.lua", log.INFO, "🔹 Full Payload Info:")
--     printtable2(payload)

    local aircraft_payload = LoGetPayloadInfo()  -- Get aircraft's payload info
    for station, payload in pairs(aircraft_payload.Stations) do
        local clsid = payload.CLSID
        local count = payload.count or 0

        -- Get weapon details from our table
        local weapon_info = weapons_data[clsid] or {name = "Unknown Weapon", weight = 0}

        log.write("Export.lua", log.INFO, string.format("🔹 Station %d: %d x %s (Weight: %.1f kg each)", station, count, weapon_info.name, weapon_info.weight))
    end





end

function LuaExportStop()
    if MySocket then
        MySocket:close()
        log.write("Export.lua", log.INFO, "✅ Socket closed")
    end
end
