import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gpt-bot")

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
VOICE_CHANNEL_ID = os.environ.get("DISCORD_VOICE_CHANNEL_ID")
TEXT_CHANNEL_ID = os.environ.get("DISCORD_TEXT_CHANNEL_ID")
ADMIN_ROLE_IDS_ENV = os.environ.get("DISCORD_ADMIN_ROLE_IDS", "")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN environment variable is required")

if not VOICE_CHANNEL_ID or not VOICE_CHANNEL_ID.isdigit():
    raise RuntimeError("DISCORD_VOICE_CHANNEL_ID environment variable must be set to a channel ID")

if not TEXT_CHANNEL_ID or not TEXT_CHANNEL_ID.isdigit():
    raise RuntimeError("DISCORD_TEXT_CHANNEL_ID environment variable must be set to a channel ID")

VOICE_CHANNEL_ID = int(VOICE_CHANNEL_ID)
TEXT_CHANNEL_ID = int(TEXT_CHANNEL_ID)
ADMIN_ROLE_IDS = {
    int(role_id.strip())
    for role_id in ADMIN_ROLE_IDS_ENV.split(",")
    if role_id.strip().isdigit()
}

DATA_FILE = Path("bot_store.json")


def ensure_data_file(path: Path) -> None:
    if not path.exists():
        path.write_text(json.dumps({"todos": {}, "reminders": []}, indent=2), encoding="utf-8")


class StoreManager:
    """Handle persistence of todo tasks and reminders."""

    def __init__(self, path: Path):
        self.path = path
        ensure_data_file(self.path)
        self._lock = asyncio.Lock()
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Failed to load data store, recreating")
            return {"todos": {}, "reminders": []}

    async def _write_locked(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self._data, file, indent=2)
        tmp_path.replace(self.path)

    async def get_tasks(self, user_id: int) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._data.get("todos", {}).get(str(user_id), []))

    async def add_task(self, user_id: int, description: str) -> Dict[str, Any]:
        async with self._lock:
            user_key = str(user_id)
            tasks = self._data.setdefault("todos", {}).setdefault(user_key, [])
            task = {
                "id": uuid.uuid4().hex,
                "description": description,
                "completed": False,
            }
            tasks.append(task)
            await self._write_locked()
            return task

    async def edit_task(self, user_id: int, task_id: str, description: str) -> bool:
        async with self._lock:
            tasks = self._data.setdefault("todos", {}).setdefault(str(user_id), [])
            for task in tasks:
                if task["id"] == task_id:
                    task["description"] = description
                    await self._write_locked()
                    return True
            return False

    async def delete_task(self, user_id: int, task_id: str) -> bool:
        async with self._lock:
            tasks = self._data.setdefault("todos", {}).setdefault(str(user_id), [])
            original_len = len(tasks)
            tasks[:] = [task for task in tasks if task["id"] != task_id]
            if len(tasks) != original_len:
                await self._write_locked()
                return True
            return False

    async def complete_task(self, user_id: int, task_id: str) -> bool:
        async with self._lock:
            tasks = self._data.setdefault("todos", {}).setdefault(str(user_id), [])
            for task in tasks:
                if task["id"] == task_id:
                    task["completed"] = True
                    await self._write_locked()
                    return True
            return False

    async def get_reminders(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._data.get("reminders", []))

    async def get_reminders_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        async with self._lock:
            reminders = [r for r in self._data.get("reminders", []) if r["user_id"] == user_id]
        return reminders

    async def add_reminder(self, reminder: Dict[str, Any]) -> None:
        async with self._lock:
            reminders = self._data.setdefault("reminders", [])
            reminders.append(reminder)
            await self._write_locked()

    async def remove_reminder(self, reminder_id: str) -> bool:
        async with self._lock:
            reminders = self._data.setdefault("reminders", [])
            original_len = len(reminders)
            reminders[:] = [r for r in reminders if r["id"] != reminder_id]
            if len(reminders) != original_len:
                await self._write_locked()
                return True
        return False


@dataclass
class Reminder:
    id: str
    user_id: int
    due_at: float
    message: str
    channel_id: Optional[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reminder":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            due_at=data["due_at"],
            message=data["message"],
            channel_id=data.get("channel_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "due_at": self.due_at,
            "message": self.message,
            "channel_id": self.channel_id,
        }


class ReminderScheduler:
    def __init__(self, bot: commands.Bot, store: StoreManager):
        self.bot = bot
        self.store = store
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        reminders = await self.store.get_reminders()
        for data in reminders:
            reminder = Reminder.from_dict(data)
            self._schedule(reminder)
        self._initialized = True
        logger.info("Loaded %s reminder(s) from disk", len(reminders))

    async def add_reminder(
        self,
        user_id: int,
        delay_seconds: float,
        message: str,
        channel_id: Optional[int],
    ) -> Reminder:
        reminder = Reminder(
            id=uuid.uuid4().hex,
            user_id=user_id,
            due_at=time.time() + delay_seconds,
            message=message,
            channel_id=channel_id,
        )
        await self.store.add_reminder(reminder.to_dict())
        self._schedule(reminder)
        return reminder

    def _schedule(self, reminder: Reminder) -> None:
        delay = max(0, reminder.due_at - time.time())
        task = asyncio.create_task(self._run_reminder(reminder, delay))
        self._tasks[reminder.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(reminder.id, None))

    async def _run_reminder(self, reminder: Reminder, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            user = self.bot.get_user(reminder.user_id) or await self.bot.fetch_user(reminder.user_id)
            content = f"⏰ تذكير: {reminder.message}"

            sent = False
            if reminder.channel_id:
                channel = self.bot.get_channel(reminder.channel_id)
                if isinstance(channel, (discord.TextChannel, discord.Thread)):
                    try:
                        await channel.send(content, allowed_mentions=discord.AllowedMentions.none())
                        sent = True
                    except discord.HTTPException as err:
                        logger.warning("Failed to send reminder to channel %s: %s", reminder.channel_id, err)

            if not sent and user:
                try:
                    await user.send(content)
                    sent = True
                except discord.HTTPException as err:
                    logger.warning("Failed to send reminder DM to user %s: %s", reminder.user_id, err)

            if not sent:
                logger.error("Reminder %s could not be delivered", reminder.id)
        except asyncio.CancelledError:
            logger.info("Reminder %s was cancelled before delivery", reminder.id)
            raise
        finally:
            await self.store.remove_reminder(reminder.id)

    async def cancel_reminder(self, reminder_id: str) -> bool:
        task = self._tasks.pop(reminder_id, None)
        removed = await self.store.remove_reminder(reminder_id)
        if task:
            task.cancel()
        return task is not None or removed


class TodoSelect(discord.ui.Select):
    def __init__(self, view: "TodoView", tasks: List[Dict[str, Any]]):
        options = self._build_options(tasks)
        super().__init__(
            placeholder="اختر مهمة لعرض تفاصيلها",
            min_values=1,
            max_values=1,
            options=options,
            disabled=not bool(options),
        )
        self.view: TodoView
        self.view = view

    @staticmethod
    def _build_options(tasks: List[Dict[str, Any]]) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = []
        for task in tasks:
            label = task["description"][:100] or "(وصف فارغ)"
            if task.get("completed"):
                label = f"✅ {label}"
            options.append(discord.SelectOption(label=label, value=task["id"]))
        return options

    def update_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        options = self._build_options(tasks)
        self.options = options
        self.disabled = not bool(options)

    async def callback(self, interaction: discord.Interaction) -> None:
        task_id = self.values[0]
        self.view.selected_task_id = task_id
        task = await self.view.store.get_task_by_id(self.view.user_id, task_id)
        if not task:
            await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)
            return
        status = "منتهية" if task.get("completed") else "قيد التنفيذ"
        await interaction.response.send_message(
            f"**المهمة:** {task['description']}\nالحالة: {status}",
            ephemeral=True,
        )


class TodoActionButton(discord.ui.Button):
    def __init__(self, label: str, style: discord.ButtonStyle, action: str):
        super().__init__(label=label, style=style)
        self.action = action

    async def callback(self, interaction: discord.Interaction) -> None:
        view: TodoView = self.view  # type: ignore[assignment]
        if not view.selected_task_id:
            await interaction.response.send_message("يرجى اختيار مهمة أولاً.", ephemeral=True)
            return
        if self.action == "complete":
            updated = await view.store.complete_task(view.user_id, view.selected_task_id)
            message = "تم تعليم المهمة كمكتملة." if updated else "تعذر إكمال المهمة."
        elif self.action == "delete":
            updated = await view.store.delete_task(view.user_id, view.selected_task_id)
            if updated:
                view.selected_task_id = None
            message = "تم حذف المهمة." if updated else "تعذر حذف المهمة."
        else:
            message = "إجراء غير معروف."

        await view.refresh()
        await interaction.response.edit_message(content=view.render_summary(), view=view)
        await interaction.followup.send(message, ephemeral=True)


class TodoView(discord.ui.View):
    def __init__(self, store: "ExtendedStoreManager", user_id: int, tasks: List[Dict[str, Any]], summary: str):
        super().__init__(timeout=180)
        self.store = store
        self.user_id = user_id
        self.selected_task_id: Optional[str] = tasks[0]["id"] if tasks else None
        self._summary = summary
        self._has_tasks = bool(tasks)
        self.select_menu = TodoSelect(self, tasks)
        self.add_item(self.select_menu)
        self.add_item(TodoActionButton("أكمل", discord.ButtonStyle.success, "complete"))
        self.add_item(TodoActionButton("احذف", discord.ButtonStyle.danger, "delete"))

    async def refresh(self) -> None:
        tasks, summary = await self.store.get_tasks_with_summary(self.user_id)
        self.select_menu.update_tasks(tasks)
        if self.selected_task_id and not any(task["id"] == self.selected_task_id for task in tasks):
            self.selected_task_id = tasks[0]["id"] if tasks else None
        self._summary = summary
        self._has_tasks = bool(tasks)

    def render_summary(self) -> str:
        if not self._has_tasks:
            return "لا توجد لديك مهام حتى الآن."
        return f"**مهامك الحالية:**\n{self._summary}"


class ExtendedStoreManager(StoreManager):
    async def get_task_by_id(self, user_id: int, task_id: str) -> Optional[Dict[str, Any]]:
        tasks = await self.get_tasks(user_id)
        for task in tasks:
            if task["id"] == task_id:
                return task
        return None

    async def get_tasks_with_summary(self, user_id: int) -> tuple[List[Dict[str, Any]], str]:
        async with self._lock:
            tasks = list(self._data.get("todos", {}).get(str(user_id), []))
        summary = self._format_tasks(tasks)
        return tasks, summary

    async def format_tasks(self, user_id: int) -> str:
        tasks = await self.get_tasks(user_id)
        return self._format_tasks(tasks)

    @staticmethod
    def _format_tasks(tasks: List[Dict[str, Any]]) -> str:
        if not tasks:
            return "لا توجد لديك مهام حتى الآن."
        lines = []
        for idx, task in enumerate(tasks, start=1):
            status = "✅" if task.get("completed") else "⬜"
            lines.append(f"{idx}. {status} {task['description']} (المعرف: {task['id']})")
        return "\n".join(lines)

    async def get_reminder_by_id(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            for reminder in self._data.get("reminders", []):
                if reminder["id"] == reminder_id:
                    return reminder
        return None

    async def get_reminders_with_summary(self, user_id: int) -> tuple[List[Dict[str, Any]], str]:
        reminders = await self.get_reminders_for_user(user_id)
        if not reminders:
            return [], "لا توجد لديك أي تذكيرات مجدولة."
        sorted_reminders = sorted(reminders, key=lambda r: r["due_at"])
        lines = []
        for idx, reminder in enumerate(sorted_reminders, start=1):
            due_time = datetime.fromtimestamp(reminder["due_at"], tz=timezone.utc)
            lines.append(
                f"{idx}. ⏰ {reminder['message']} — الموعد {due_time.isoformat()} (المعرف: {reminder['id']})"
            )
        return sorted_reminders, "\n".join(lines)


class ReminderSelect(discord.ui.Select):
    def __init__(self, view: "ReminderView", reminders: List[Dict[str, Any]]):
        options = self._build_options(reminders)
        super().__init__(
            placeholder="اختر تذكيراً لإدارته",
            min_values=1,
            max_values=1,
            options=options,
            disabled=not bool(options),
        )
        self.view: ReminderView
        self.view = view

    @staticmethod
    def _build_options(reminders: List[Dict[str, Any]]) -> List[discord.SelectOption]:
        options: List[discord.SelectOption] = []
        for reminder in reminders:
            due_time = datetime.fromtimestamp(reminder["due_at"], tz=timezone.utc)
            label = due_time.strftime("%Y-%m-%d %H:%M UTC")
            description = reminder["message"][:100] or "(لا توجد رسالة)"
            options.append(discord.SelectOption(label=label, description=description, value=reminder["id"]))
        return options

    def update_reminders(self, reminders: List[Dict[str, Any]]) -> None:
        options = self._build_options(reminders)
        self.options = options
        self.disabled = not bool(options)

    async def callback(self, interaction: discord.Interaction) -> None:
        reminder_id = self.values[0]
        self.view.selected_reminder_id = reminder_id
        reminder = await self.view.store.get_reminder_by_id(reminder_id)
        if not reminder:
            await interaction.response.send_message("تعذر العثور على التذكير.", ephemeral=True)
        return
        due_time = datetime.fromtimestamp(reminder["due_at"], tz=timezone.utc)
        await interaction.response.send_message(
            f"**التذكير:** {reminder['message']}\nالموعد: {due_time.isoformat()}",
            ephemeral=True,
        )


class ReminderCancelButton(discord.ui.Button):
    def __init__(self) -> None:
        super().__init__(label="ألغِ التذكير", style=discord.ButtonStyle.danger)

    async def callback(self, interaction: discord.Interaction) -> None:
        view: ReminderView = self.view  # type: ignore[assignment]
        if not view.selected_reminder_id:
            await interaction.response.send_message("يرجى اختيار تذكير أولاً.", ephemeral=True)
            return
        reminder_id = view.selected_reminder_id
        cancelled = await view.scheduler.cancel_reminder(reminder_id)
        await view.refresh()
        await interaction.response.edit_message(content=view.render_summary(), view=view)
        status = "تم إلغاء التذكير." if cancelled else "تعذر العثور على التذكير."
        await interaction.followup.send(status, ephemeral=True)


class ReminderView(discord.ui.View):
    def __init__(
        self,
        store: "ExtendedStoreManager",
        scheduler: ReminderScheduler,
        user_id: int,
        reminders: List[Dict[str, Any]],
        summary: str,
    ) -> None:
        super().__init__(timeout=180)
        self.store = store
        self.scheduler = scheduler
        self.user_id = user_id
        self.selected_reminder_id: Optional[str] = reminders[0]["id"] if reminders else None
        self._summary = summary
        self._has_reminders = bool(reminders)
        self.select_menu = ReminderSelect(self, reminders)
        self.add_item(self.select_menu)
        self.add_item(ReminderCancelButton())

    async def refresh(self) -> None:
        reminders, summary = await self.store.get_reminders_with_summary(self.user_id)
        self.select_menu.update_reminders(reminders)
        if self.selected_reminder_id and not any(r["id"] == self.selected_reminder_id for r in reminders):
            self.selected_reminder_id = reminders[0]["id"] if reminders else None
        self._summary = summary
        self._has_reminders = bool(reminders)

    def render_summary(self) -> str:
        if not self._has_reminders:
            return "لا توجد لديك أي تذكيرات مجدولة."
        return f"**تذكيراتك:**\n{self._summary}"


intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)
store = ExtendedStoreManager(DATA_FILE)
scheduler = ReminderScheduler(bot, store)

SMART_SHORTCUTS: Dict[str, Dict[str, Any]] = {
    "focus_25": {
        "title": "جلسة تركيز ٢٥ دقيقة",
        "description": "يضبط تذكيراً بعد ٢٥ دقيقة للاستراحة من جلسة التركيز.",
        "reminder": {
            "minutes": 25,
            "message": "انتهت جلسة التركيز! خذ استراحة قصيرة وعاود النشاط.",
        },
    },
    "hydrate_hourly": {
        "title": "تذكير شرب الماء كل ساعة",
        "description": "يضبط تذكيراً بعد ٦٠ دقيقة لشرب الماء والحفاظ على نشاطك.",
        "reminder": {
            "minutes": 60,
            "message": "حان وقت شرب الماء لتجديد نشاطك!",
        },
    },
    "daily_kickoff": {
        "title": "انطلاقة يومية سريعة",
        "description": "يضيف حزمة مهام صباحية أساسية للتخطيط ليومك.",
        "tasks": [
            "مراجعة الأهداف الرئيسية لليوم",
            "تنظيم المهام حسب الأولوية",
            "التأكد من مواعيد الاجتماعات",
        ],
    },
}

SHORTCUT_CHOICES: List[app_commands.Choice[str]] = [
    app_commands.Choice(name=data["title"], value=code)
    for code, data in SMART_SHORTCUTS.items()
]


def build_shortcut_help_text() -> str:
    lines = []
    for code, data in SMART_SHORTCUTS.items():
        lines.append(f"`{code}` — {data['title']}: {data['description']}")
    return "\n".join(lines)


async def execute_smart_shortcut(user_id: int, code: str, channel_id: Optional[int]) -> tuple[str, bool]:
    definition = SMART_SHORTCUTS.get(code)
    if not definition:
        raise KeyError(code)

    lines = [f"✅ تم تنفيذ اختصار **{definition['title']}**"]
    tasks_added = False

    tasks_to_add = definition.get("tasks", [])
    if tasks_to_add:
        for task_description in tasks_to_add:
            await store.add_task(user_id, task_description)
        tasks_added = True
        lines.append(f"• تمت إضافة {len(tasks_to_add)} مهمة جاهزة إلى قائمتك.")

    reminder_info = definition.get("reminder")
    if reminder_info:
        reminder = await scheduler.add_reminder(
            user_id,
            reminder_info["minutes"] * 60,
            reminder_info["message"],
            channel_id,
        )
        lines.append(
            f"• تم ضبط تذكير بعد {reminder_info['minutes']} دقيقة/دقائق (المعرف: {reminder.id})."
        )

    return "\n".join(lines), tasks_added


def is_admin() -> app_commands.Check:
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.user:
            return False
        if isinstance(interaction.user, discord.Member):
            if ADMIN_ROLE_IDS and any(role.id in ADMIN_ROLE_IDS for role in interaction.user.roles):
                return True
        if not ADMIN_ROLE_IDS:
            guild = interaction.guild
            if guild:
                member = guild.get_member(interaction.user.id)
                if member and any(role.permissions.administrator for role in member.roles):
                    return True
        return False

    return app_commands.check(predicate)


@bot.event
async def on_ready() -> None:
    await scheduler.initialize()
    await bot.tree.sync()
    logger.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")

    voice_channel = bot.get_channel(VOICE_CHANNEL_ID)
    if isinstance(voice_channel, discord.VoiceChannel):
        if voice_channel.guild.voice_client is None:
            try:
                await voice_channel.connect(reconnect=True)
                logger.info("Connected to voice channel %s", voice_channel.name)
            except discord.ClientException as err:
                logger.warning("Failed to connect to voice channel: %s", err)
    else:
        logger.error("Configured voice channel ID %s is invalid", VOICE_CHANNEL_ID)

    text_channel = bot.get_channel(TEXT_CHANNEL_ID)
    if isinstance(text_channel, (discord.TextChannel, discord.Thread)):
        try:
            await text_channel.send(
                "الروبوت جاهز للعمل!", allowed_mentions=discord.AllowedMentions.none()
            )
        except discord.HTTPException as err:
            logger.warning("Failed to send readiness message: %s", err)


@bot.tree.command(name="broadcast", description="إرسال تنبيه إلى القناة النصية المحددة")
@is_admin()
async def broadcast(interaction: discord.Interaction, message: str) -> None:
    channel = bot.get_channel(TEXT_CHANNEL_ID)
    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        await interaction.response.send_message("تعذر العثور على القناة النصية المحددة.", ephemeral=True)
        return
    await channel.send(message, allowed_mentions=discord.AllowedMentions.none())
    await interaction.response.send_message("تم إرسال التنبيه.", ephemeral=True)


@bot.tree.command(name="play_audio", description="تشغيل الصوت في القناة الصوتية المتصل بها")
@is_admin()
async def play_audio(interaction: discord.Interaction, source_url: str) -> None:
    voice_client = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not voice_client:
        await interaction.response.send_message("الروبوت غير متصل بقناة صوتية.", ephemeral=True)
        return
    if voice_client.is_playing():
        voice_client.stop()
    try:
        audio_source = discord.FFmpegPCMAudio(source_url)
    except Exception as err:  # pragma: no cover - depends on ffmpeg availability
        await interaction.response.send_message(f"تعذر بدء التشغيل: {err}", ephemeral=True)
        return

    voice_client.play(audio_source)
    await interaction.response.send_message("تم بدء تشغيل الصوت.", ephemeral=True)


@bot.tree.command(name="stop_audio", description="إيقاف تشغيل الصوت في القناة الصوتية")
@is_admin()
async def stop_audio(interaction: discord.Interaction) -> None:
    voice_client = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    if not voice_client or not voice_client.is_playing():
        await interaction.response.send_message("لا يوجد صوت قيد التشغيل حالياً.", ephemeral=True)
        return
    voice_client.stop()
    await interaction.response.send_message("تم إيقاف تشغيل الصوت.", ephemeral=True)


todo_group = app_commands.Group(name="todo", description="إدارة قائمة مهامك")


@todo_group.command(name="add", description="إضافة مهمة إلى قائمتك الشخصية")
async def todo_add(interaction: discord.Interaction, description: str) -> None:
    await store.add_task(interaction.user.id, description)
    tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
    view = TodoView(store, interaction.user.id, tasks, summary)
    await interaction.response.send_message(
        content=view.render_summary(),
        view=view,
        ephemeral=True,
    )


@todo_group.command(name="edit", description="تعديل مهمة موجودة باستخدام معرفها")
async def todo_edit(interaction: discord.Interaction, task_id: str, description: str) -> None:
    updated = await store.edit_task(interaction.user.id, task_id, description)
    if updated:
        await interaction.response.send_message("تم تحديث المهمة.", ephemeral=True)
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="delete", description="حذف مهمة باستخدام معرفها")
async def todo_delete(interaction: discord.Interaction, task_id: str) -> None:
    deleted = await store.delete_task(interaction.user.id, task_id)
    if deleted:
        await interaction.response.send_message("تم حذف المهمة.", ephemeral=True)
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="complete", description="تعليم مهمة كمكتملة")
async def todo_complete(interaction: discord.Interaction, task_id: str) -> None:
    completed = await store.complete_task(interaction.user.id, task_id)
    if completed:
        await interaction.response.send_message("تم تعليم المهمة كمكتملة.", ephemeral=True)
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="list", description="عرض مهامك الحالية")
async def todo_list(interaction: discord.Interaction) -> None:
    tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
    view = TodoView(store, interaction.user.id, tasks, summary)
    await interaction.response.send_message(view.render_summary(), view=view, ephemeral=True)


bot.tree.add_command(todo_group)


remind_group = app_commands.Group(name="remind", description="أوامر التذكير")


@remind_group.command(name="me", description="جدولة تذكير")
async def remind_me(
    interaction: discord.Interaction,
    message: str,
    minutes: app_commands.Range[int, 1, 7 * 24 * 60],
    channel: Optional[discord.TextChannel] = None,
) -> None:
    delay_seconds = minutes * 60
    reminder = await scheduler.add_reminder(
        interaction.user.id,
        delay_seconds,
        message,
        channel.id if channel else None,
    )
    await interaction.response.send_message(
        f"تم ضبط التذكير بعد {minutes} دقيقة/دقائق. المعرف: {reminder.id}",
        ephemeral=True,
    )


@remind_group.command(name="list", description="عرض تذكيراتك أو إلغاؤها")
async def remind_list(interaction: discord.Interaction) -> None:
    reminders, summary = await store.get_reminders_with_summary(interaction.user.id)
    view = ReminderView(store, scheduler, interaction.user.id, reminders, summary)
    await interaction.response.send_message(view.render_summary(), view=view, ephemeral=True)


bot.tree.add_command(remind_group)


shortcut_group = app_commands.Group(name="shortcut", description="اختصارات ذكية جاهزة")


@shortcut_group.command(name="run", description="تشغيل اختصار ذكي محدد")
@app_commands.describe(
    shortcut="اختر الاختصار المطلوب",
    channel="القناة التي سيصلها التذكير (اختياري)",
)
@app_commands.choices(shortcut=SHORTCUT_CHOICES)
async def shortcut_run(
    interaction: discord.Interaction,
    shortcut: app_commands.Choice[str],
    channel: Optional[discord.TextChannel] = None,
) -> None:
    await interaction.response.defer(ephemeral=True)
    try:
        message, tasks_added = await execute_smart_shortcut(
            interaction.user.id,
            shortcut.value,
            channel.id if channel else interaction.channel_id,
        )
    except KeyError:
        await interaction.followup.send("الاختصار المحدد غير معروف.", ephemeral=True)
        return

    if tasks_added:
        tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
        view = TodoView(store, interaction.user.id, tasks, summary)
        await interaction.followup.send(
            f"{message}\n\n{view.render_summary()}",
            view=view,
            ephemeral=True,
        )
    else:
        await interaction.followup.send(message, ephemeral=True)


@shortcut_group.command(name="list", description="عرض الاختصارات الذكية المتاحة")
async def shortcut_list(interaction: discord.Interaction) -> None:
    help_text = build_shortcut_help_text()
    await interaction.response.send_message(
        f"**الاختصارات المتاحة:**\n{help_text}",
        ephemeral=True,
    )


bot.tree.add_command(shortcut_group)


@bot.command(name="shortcut", help="تشغيل اختصار ذكي محدد (مثال: !shortcut focus_25)")
async def shortcut_prefix(ctx: commands.Context, code: str) -> None:
    try:
        message, _ = await execute_smart_shortcut(
            ctx.author.id,
            code.lower(),
            ctx.channel.id if isinstance(ctx.channel, (discord.TextChannel, discord.Thread)) else None,
        )
    except KeyError:
        help_text = build_shortcut_help_text()
        await ctx.reply(
            f"الاختصار غير معروف. استخدم أحد الأكواد التالية:\n{help_text}",
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    await ctx.reply(
        message,
        mention_author=False,
        allowed_mentions=discord.AllowedMentions.none(),
    )


@bot.command(name="shortcuts", help="عرض الاختصارات الذكية المتاحة")
async def shortcuts_prefix(ctx: commands.Context) -> None:
    help_text = build_shortcut_help_text()
    await ctx.reply(
        f"**الاختصارات المتاحة:**\n{help_text}",
        mention_author=False,
        allowed_mentions=discord.AllowedMentions.none(),
    )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
