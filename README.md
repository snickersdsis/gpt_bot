# GPT Discord Bot

This project hosts a feature-rich [discord.py](https://discordpy.readthedocs.io/en/stable/) bot that manages voice playback, administrator announcements, personal to-do lists, and reminders.

## Requirements

- Python 3.10+
- `discord.py` (install with `pip install -U discord.py`)
- FFmpeg available on the host system for audio playback commands

## Configuration

Configure the bot via environment variables before running it:

| Variable | Description |
| --- | --- |
| `DISCORD_BOT_TOKEN` | Bot token from the Discord Developer Portal. |
| `DISCORD_VOICE_CHANNEL_ID` | Numeric ID of the voice channel the bot should join on startup. |
| `DISCORD_TEXT_CHANNEL_ID` | Numeric ID of the text channel used for readiness pings and broadcasts. |
| `DISCORD_ADMIN_ROLE_IDS` | (Optional) Comma-separated list of role IDs that are allowed to run admin commands. If omitted, server administrators are allowed automatically. |

### Keeping secrets safe (Railway & local)

- **Railway:** store the Discord token and channel IDs as project variables so they never appear in source control:
  ```bash
  railway variables set DISCORD_BOT_TOKEN=... DISCORD_VOICE_CHANNEL_ID=... DISCORD_TEXT_CHANNEL_ID=...
  ```
  Railway exposes those variables to the container automatically, so `main.py` reads them without any hard-coded secrets.
- **Local development:** export the same variables in your shell (or use a `.env` file loaded with [direnv](https://direnv.net/) / similar tooling) before launching the bot.

## Running the Bot

1. Install dependencies: `pip install -U discord.py`
2. Export the environment variables listed above.
3. Launch the bot:
   ```bash
   python main.py
   ```

On startup the bot will connect to the configured voice channel and announce its readiness in the configured text channel. Slash commands under `/todo` manage per-user task lists and display interactive dropdowns/buttons for quick actions (all bot responses and support text appear in Arabic). Use the new `/todo priority`, `/todo schedule`, `/todo prioritize`, `/todo stats`, `/todo advise`, and `/todo focus` commands to control priorities, due dates, smart insights, and focus sessions. Reminders can be scheduled with `/remind me` or the natural-language `/remind smart`, and admin-only commands manage broadcasts and voice playback.

### Smart shortcuts (اختصارات ذكية)

Use the `/shortcut` slash commands or the `!shortcut`/`!shortcuts` text commands to trigger ready-made workflows:

- `focus_25` — creates a 25-minute focus reminder that pings you in Arabic when it completes.
- `hydrate_hourly` — schedules a hydration reminder after 60 minutes.
- `daily_kickoff` — drops a pre-filled morning checklist directly into your personal to-do list.
- `power_morning` — prepares a high-impact morning routine with priority-ranked tasks and a progress reminder.
- `wrap_day` — helps you close the day with review tasks and a follow-up reminder for tomorrow.

Run `/shortcut list` (or `!shortcuts`) to view the available codes from within Discord.

Data (to-do tasks and scheduled reminders) is persisted to `bot_store.json` in the project root so it survives restarts.
