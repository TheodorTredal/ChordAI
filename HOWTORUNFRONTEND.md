For å kjøre frontend inne på klusteret.

1. På c0-0 kjør:
    npm run dev -- --host 0.0.0.0

2. lokalt i terminalen på PC'en din, kjør:
    ssh -L 8000:c0-0:3000 abc123@ificluster.ifi.uit.no

3. på din lokale PC gå til denne lenken:
    http://localhost:8000
