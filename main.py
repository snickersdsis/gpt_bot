import asyncio
import json
import logging
import os
import re
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


_ARABIC_DURATION_FORMS = {
    "second": {"singular": "ثانية", "dual": "ثانيتين", "plural": "ثوان"},
    "minute": {"singular": "دقيقة", "dual": "دقيقتين", "plural": "دقائق"},
    "hour": {"singular": "ساعة", "dual": "ساعتين", "plural": "ساعات"},
    "day": {"singular": "يوم", "dual": "يومين", "plural": "أيام"},
}


def _format_arabic_unit(value: int, forms: Dict[str, str]) -> str:
    if value == 1:
        return f"{value} {forms['singular']}"
    if value == 2:
        return f"{value} {forms['dual']}"
    if 3 <= value <= 10:
        return f"{value} {forms['plural']}"
    return f"{value} {forms['singular']}"


def describe_duration(seconds: float) -> str:
    seconds = int(abs(seconds))
    if seconds < 1:
        return "لحظات"

    parts: List[str] = []
    units = [
        (24 * 3600, _ARABIC_DURATION_FORMS["day"]),
        (3600, _ARABIC_DURATION_FORMS["hour"]),
        (60, _ARABIC_DURATION_FORMS["minute"]),
        (1, _ARABIC_DURATION_FORMS["second"]),
    ]

    remaining = seconds
    for unit_seconds, forms in units:
        if remaining < unit_seconds:
            continue
        value, remaining = divmod(remaining, unit_seconds)
        if value:
            parts.append(_format_arabic_unit(value, forms))
        if len(parts) == 2:
            break

    if not parts:
        parts.append("لحظات")
    return " و ".join(parts)


def format_due_relative(due_at: Optional[float]) -> Optional[str]:
    if not due_at:
        return None
    delta = due_at - time.time()
    if abs(delta) < 5:
        return "الآن"
    description = describe_duration(delta)
    if delta > 0:
        return f"بعد {description}"
    return f"متأخر منذ {description}"


def format_due_absolute(due_at: Optional[float]) -> Optional[str]:
    if not due_at:
        return None
    due_time = datetime.fromtimestamp(due_at, tz=timezone.utc)
    return due_time.strftime("%Y-%m-%d %H:%M UTC")


_NATURAL_KEYWORDS = {
    "ساعتين": 2 * 3600,
    "دقيقتين": 2 * 60,
    "دقيقتان": 2 * 60,
    "ثانيتين": 2,
    "ثانيتان": 2,
    "يومين": 2 * 24 * 3600,
    "غداً": 24 * 3600,
    "غدا": 24 * 3600,
}


def parse_natural_delay(expression: str) -> Optional[int]:
    text = expression.strip().lower()
    if not text:
        return None

    total_seconds = 0
    pattern = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*(يوم|أيام|ايام|ساعة|ساعات|س|دقيقة|دقائق|د|ثانية|ثوان|ث)"
    )
    for match in pattern.finditer(text):
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2)
        if unit in {"يوم", "أيام", "ايام"}:
            multiplier = 24 * 3600
        elif unit in {"ساعة", "ساعات", "س"}:
            multiplier = 3600
        elif unit in {"دقيقة", "دقائق", "د"}:
            multiplier = 60
        else:
            multiplier = 1
        total_seconds += int(value * multiplier)

    for keyword, seconds_value in _NATURAL_KEYWORDS.items():
        if keyword in text:
            total_seconds += seconds_value

    if "نصف ساعة" in text:
        total_seconds += 30 * 60
    if "ربع ساعة" in text:
        total_seconds += 15 * 60
    if "ثلاثة ارباع" in text or "ثلاثة أرباع" in text:
        total_seconds += int(45 * 60)

    if total_seconds <= 0:
        return None
    return total_seconds


PRIORITY_BADGES = {
    1: "🔥",
    2: "⚡",
    3: "✅",
    4: "🕒",
    5: "🌙",
}


def normalize_priority(priority: Optional[int]) -> int:
    if priority is None:
        return 3
    return max(1, min(5, priority))


def describe_priority(priority: int) -> str:
    priority = normalize_priority(priority)
    if priority == 1:
        return "عالية جداً"
    if priority == 2:
        return "عالية"
    if priority == 3:
        return "متوسطة"
    if priority == 4:
        return "منخفضة"
    return "منخفضة جداً"


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

    async def add_task(
        self,
        user_id: int,
        description: str,
        *,
        priority: Optional[int] = None,
        due_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            user_key = str(user_id)
            tasks = self._data.setdefault("todos", {}).setdefault(user_key, [])
            now_ts = time.time()
            task = {
                "id": uuid.uuid4().hex,
                "description": description,
                "completed": False,
                "created_at": now_ts,
                "priority": normalize_priority(priority),
            }
            if due_at:
                task["due_at"] = due_at
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
                    task["completed_at"] = time.time()
                    await self._write_locked()
                    return True
            return False

    async def update_task_priority(self, user_id: int, task_id: str, priority: int) -> bool:
        async with self._lock:
            tasks = self._data.setdefault("todos", {}).setdefault(str(user_id), [])
            for task in tasks:
                if task["id"] == task_id:
                    task["priority"] = normalize_priority(priority)
                    await self._write_locked()
                    return True
        return False

    async def update_task_due(self, user_id: int, task_id: str, due_at: Optional[float]) -> bool:
        async with self._lock:
            tasks = self._data.setdefault("todos", {}).setdefault(str(user_id), [])
            for task in tasks:
                if task["id"] == task_id:
                    if due_at is None:
                        task.pop("due_at", None)
                    else:
                        task["due_at"] = due_at
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

    async def sort_tasks_by_priority(self, user_id: int) -> bool:
        async with self._lock:
            tasks = self._data.setdefault("todos", {}).setdefault(str(user_id), [])
            if not tasks:
                return False

            def sort_key(task: Dict[str, Any]) -> tuple[Any, Any, Any]:
                priority = task.get("priority", 3)
                due_at = task.get("due_at")
                created_at = task.get("created_at", 0)
                return (priority, due_at if due_at is not None else float("inf"), created_at)

            tasks.sort(key=sort_key)
            await self._write_locked()
            return True

    async def get_task_stats(self, user_id: int) -> Dict[str, int]:
        async with self._lock:
            tasks = list(self._data.get("todos", {}).get(str(user_id), []))
        total = len(tasks)
        completed = sum(1 for task in tasks if task.get("completed"))
        overdue = sum(
            1
            for task in tasks
            if not task.get("completed") and task.get("due_at") and task["due_at"] < time.time()
        )
        high_priority = sum(1 for task in tasks if task.get("priority", 3) <= 2)
        return {
            "total": total,
            "completed": completed,
            "pending": total - completed,
            "overdue": overdue,
            "high_priority": high_priority,
        }

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
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reminder":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            due_at=data["due_at"],
            message=data["message"],
            channel_id=data.get("channel_id"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "user_id": self.user_id,
            "due_at": self.due_at,
            "message": self.message,
            "channel_id": self.channel_id,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


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
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Reminder:
        reminder = Reminder(
            id=uuid.uuid4().hex,
            user_id=user_id,
            due_at=time.time() + delay_seconds,
            message=message,
            channel_id=channel_id,
            metadata=metadata,
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
            icon = "⏰"
            prefix = "تذكير"
            raw_message = False
            if reminder.metadata:
                icon = reminder.metadata.get("icon", icon)
                prefix = reminder.metadata.get("prefix", prefix)
                raw_message = reminder.metadata.get("raw_message", raw_message)
            if raw_message:
                content = f"{icon} {reminder.message}"
            else:
                content = f"{icon} {prefix}: {reminder.message}"

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


class ProductivityAdvisor:
    def __init__(self, store: ExtendedStoreManager):
        self.store = store

    async def build_report(self, user_id: int) -> str:
        stats = await self.store.get_task_stats(user_id)
        tasks = await self.store.get_tasks(user_id)
        if not tasks:
            return "قائمتك فارغة حالياً — أضف مهمة جديدة وابدأ يومك بقوة!"

        lines = [
            "**نظرة سريعة على إنتاجيتك:**",
            f"• إجمالي المهام: {stats['total']}",
            f"• المهام المنجزة: {stats['completed']}",
            f"• المهام المتبقية: {stats['pending']}",
        ]
        if stats["overdue"]:
            lines.append(f"• مهام متأخرة تحتاج إلى اهتمام عاجل: {stats['overdue']}")
        if stats["high_priority"]:
            lines.append(f"• مهام عالية الأهمية قيد الانتظار: {stats['high_priority']}")

        pending_tasks = [
            task
            for task in tasks
            if not task.get("completed")
        ]
        if pending_tasks:
            high_priority = [
                task
                for task in pending_tasks
                if normalize_priority(task.get("priority")) <= 2
            ]
            if high_priority:
                lines.append("\n**أولوياتك القصوى:**")
                for task in high_priority[:3]:
                    due_text = format_due_relative(task.get("due_at"))
                    badge = PRIORITY_BADGES.get(normalize_priority(task.get("priority")), "⚡")
                    extra = f" — {due_text}" if due_text else ""
                    lines.append(f"{badge} {task['description']}{extra}")
            upcoming = sorted(
                [task for task in pending_tasks if task.get("due_at")],
                key=lambda item: item["due_at"],
            )
            if upcoming:
                lines.append("\n**المواعيد الأقرب:**")
                for task in upcoming[:3]:
                    due_text = format_due_relative(task.get("due_at")) or "في أقرب وقت"
                    lines.append(f"⏳ {task['description']} — {due_text}")

        if stats["completed"]:
            latest_completion = max(
                (task.get("completed_at", 0) for task in tasks if task.get("completed")),
                default=0,
            )
            if latest_completion:
                ago_text = describe_duration(time.time() - latest_completion)
                lines.append(f"\nآخر إنجاز كان منذ {ago_text}. استمر بالزخم نفسه!")

        if stats["pending"] > stats["completed"]:
            lines.append("🔁 جرّب تقسيم المهام الكبيرة إلى خطوات صغيرة لتسريع التقدم.")
        else:
            lines.append("🏆 عمل رائع! حافظ على هذا الإيقاع وحدد مهمة جديدة لتستبق احتياجاتك.")

        return "\n".join(lines)


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
            priority_value = normalize_priority(task.get("priority"))
            badge = PRIORITY_BADGES.get(priority_value, "✅")
            label = f"{badge} {label}"[:100]
            due_info = format_due_relative(task.get("due_at"))
            options.append(
                discord.SelectOption(
                    label=label,
                    value=task["id"],
                    description=due_info[:100] if due_info else None,
                )
            )
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
            priority_value = normalize_priority(task.get("priority"))
            priority_badge = PRIORITY_BADGES.get(priority_value, "✅")
            priority_text = describe_priority(priority_value)
            due_relative = format_due_relative(task.get("due_at"))
            due_absolute = format_due_absolute(task.get("due_at"))
            due_clause = ""
            if due_relative:
                if due_absolute:
                    due_clause = f" — الموعد {due_relative} ({due_absolute})"
                else:
                    due_clause = f" — الموعد {due_relative}"
            lines.append(
                f"{idx}. {status} {priority_badge} {task['description']} (المعرف: {task['id']})"
                f" — أولوية {priority_text}{due_clause}"
            )
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
advisor = ProductivityAdvisor(store)

SMART_SHORTCUTS: Dict[str, Dict[str, Any]] = {
    "focus_25": {
        "title": "جلسة تركيز ٢٥ دقيقة",
        "description": "يضبط تذكيراً بعد ٢٥ دقيقة للاستراحة من جلسة التركيز.",
        "reminder": {
            "minutes": 25,
            "message": "انتهت جلسة التركيز! خذ استراحة قصيرة وعاود النشاط.",
            "icon": "🔥",
            "prefix": "جلسة التركيز",
        },
    },
    "hydrate_hourly": {
        "title": "تذكير شرب الماء كل ساعة",
        "description": "يضبط تذكيراً بعد ٦٠ دقيقة لشرب الماء والحفاظ على نشاطك.",
        "reminder": {
            "minutes": 60,
            "message": "حان وقت شرب الماء لتجديد نشاطك!",
            "icon": "💧",
            "prefix": "صحتك",
        },
    },
    "daily_kickoff": {
        "title": "انطلاقة يومية سريعة",
        "description": "يضيف حزمة مهام صباحية أساسية للتخطيط ليومك.",
        "tasks": [
            {"description": "مراجعة الأهداف الرئيسية لليوم", "priority": 1},
            {"description": "تنظيم المهام حسب الأولوية", "priority": 2},
            {"description": "التأكد من مواعيد الاجتماعات", "priority": 2},
        ],
    },
    "power_morning": {
        "title": "روتين صباحي خارق",
        "description": "يحضر جدولاً متكاملاً مع مهام عالية الأولوية وتذكير تنشيطي.",
        "tasks": [
            {"description": "مراجعة بريد العمل العاجل", "priority": 1},
            {"description": "تحديد أهم ثلاث نتائج لليوم", "priority": 1},
            {"description": "تحضير قائمة الاجتماعات والأسئلة", "priority": 2},
        ],
        "reminder": {
            "minutes": 90,
            "message": "حان وقت التحقق من تقدمك في نتائج اليوم الرئيسية!",
            "icon": "🚀",
            "prefix": "مراجعة التقدم",
        },
    },
    "wrap_day": {
        "title": "إغلاق اليوم باحتراف",
        "description": "يساعدك على مراجعة الإنجازات وضبط تذكير للغد.",
        "tasks": [
            {"description": "تلخيص أهم ما تم إنجازه اليوم", "priority": 2},
            {"description": "تحديد أهم أولويات الغد", "priority": 1},
            {"description": "تحديث لوحة المتابعة", "priority": 3},
        ],
        "reminder": {
            "minutes": 12 * 60,
            "message": "تذكر مراجعة خطة الغد قبل بداية الدوام.",
            "icon": "🌙",
            "prefix": "ختام اليوم",
        },
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
        for task_definition in tasks_to_add:
            if isinstance(task_definition, str):
                await store.add_task(user_id, task_definition)
            else:
                description = task_definition.get("description", "مهمة بدون عنوان")
                priority = task_definition.get("priority")
                due_minutes = task_definition.get("due_minutes")
                due_at = (
                    time.time() + due_minutes * 60 if due_minutes is not None else None
                )
                await store.add_task(
                    user_id,
                    description,
                    priority=priority,
                    due_at=due_at,
                )
        tasks_added = True
        lines.append(f"• تمت إضافة {len(tasks_to_add)} مهمة جاهزة إلى قائمتك.")

    reminder_info = definition.get("reminder")
    if reminder_info:
        metadata = {
            "icon": reminder_info.get("icon", "⏰"),
            "prefix": reminder_info.get("prefix", "تذكير"),
            "raw_message": reminder_info.get("raw_message", False),
        }
        reminder = await scheduler.add_reminder(
            user_id,
            reminder_info["minutes"] * 60,
            reminder_info["message"],
            channel_id,
            metadata=metadata,
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
@app_commands.describe(
    description="وصف المهمة",
    priority="درجة الأهمية من ١ (عالية جداً) إلى ٥ (منخفضة جداً)",
    due_in_minutes="عدد الدقائق قبل استحقاق المهمة (اختياري)",
    auto_remind="هل تريد إنشاء تذكير تلقائي للموعد؟",
    remind_channel="القناة التي سيصلها التذكير (يتم استخدام القناة الحالية افتراضياً)",
)
async def todo_add(
    interaction: discord.Interaction,
    description: str,
    priority: Optional[app_commands.Range[int, 1, 5]] = None,
    due_in_minutes: Optional[app_commands.Range[int, 5, 60 * 24 * 30]] = None,
    auto_remind: bool = True,
    remind_channel: Optional[discord.TextChannel] = None,
) -> None:
    due_at = None
    reminder_note = ""
    if due_in_minutes:
        due_at = time.time() + (due_in_minutes * 60)
    task = await store.add_task(
        interaction.user.id,
        description,
        priority=priority,
        due_at=due_at,
    )

    if due_at and auto_remind:
        reminder_message = f"{description}"
        reminder = await scheduler.add_reminder(
            interaction.user.id,
            due_in_minutes * 60,
            reminder_message,
            (remind_channel.id if remind_channel else interaction.channel_id),
            metadata={
                "icon": "⏳",
                "prefix": "الموعد",
            },
        )
        reminder_note = (
            f"\n🔔 تم إنشاء تذكير آلي (المعرف: {reminder.id}) وسيتم تنبيهك عند الموعد."
        )

    tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
    view = TodoView(store, interaction.user.id, tasks, summary)
    priority_text = describe_priority(task.get("priority", 3))
    due_text = format_due_relative(task.get("due_at"))
    due_sentence = f" الموعد {due_text}." if due_text else ""
    await interaction.response.send_message(
        content=(
            f"تمت إضافة المهمة الجديدة بأولوية {priority_text}.{due_sentence}{reminder_note}\n\n"
            f"{view.render_summary()}"
        ),
        view=view,
        ephemeral=True,
    )


@todo_group.command(name="edit", description="تعديل مهمة موجودة باستخدام معرفها")
async def todo_edit(interaction: discord.Interaction, task_id: str, description: str) -> None:
    updated = await store.edit_task(interaction.user.id, task_id, description)
    if updated:
        tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
        view = TodoView(store, interaction.user.id, tasks, summary)
        await interaction.response.send_message(
            f"تم تحديث المهمة.\n\n{view.render_summary()}",
            view=view,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="priority", description="تعديل أولوية مهمة")
@app_commands.describe(
    task_id="معرف المهمة",
    priority="درجة الأهمية الجديدة من ١ (عالية جداً) إلى ٥ (منخفضة جداً)",
)
async def todo_priority(
    interaction: discord.Interaction,
    task_id: str,
    priority: app_commands.Range[int, 1, 5],
) -> None:
    updated = await store.update_task_priority(interaction.user.id, task_id, priority)
    if updated:
        tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
        view = TodoView(store, interaction.user.id, tasks, summary)
        await interaction.response.send_message(
            f"تم تحديث أولوية المهمة إلى {describe_priority(priority)}.\n\n{view.render_summary()}",
            view=view,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="delete", description="حذف مهمة باستخدام معرفها")
async def todo_delete(interaction: discord.Interaction, task_id: str) -> None:
    deleted = await store.delete_task(interaction.user.id, task_id)
    if deleted:
        tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
        view = TodoView(store, interaction.user.id, tasks, summary)
        await interaction.response.send_message(
            f"تم حذف المهمة.\n\n{view.render_summary()}",
            view=view,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="schedule", description="تحديد موعد لمهمة")
@app_commands.describe(
    task_id="معرف المهمة",
    due_in_minutes="عدد الدقائق حتى موعد المهمة (اختياري)",
    remove_due="إزالة الموعد الحالي للمهمة",
    auto_remind="إنشاء تذكير تلقائي عند تحديد موعد",
    remind_channel="القناة التي سيصلها التذكير",
)
async def todo_schedule(
    interaction: discord.Interaction,
    task_id: str,
    due_in_minutes: Optional[app_commands.Range[int, 5, 60 * 24 * 30]] = None,
    remove_due: bool = False,
    auto_remind: bool = True,
    remind_channel: Optional[discord.TextChannel] = None,
) -> None:
    if remove_due:
        updated = await store.update_task_due(interaction.user.id, task_id, None)
        if updated:
            tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
            view = TodoView(store, interaction.user.id, tasks, summary)
            await interaction.response.send_message(
                "تمت إزالة موعد المهمة.\n\n" + view.render_summary(),
                view=view,
                ephemeral=True,
            )
        else:
            await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)
        return

    if not due_in_minutes:
        await interaction.response.send_message(
            "يرجى تحديد عدد الدقائق أو اختيار إزالة الموعد.",
            ephemeral=True,
        )
        return

    due_at = time.time() + (due_in_minutes * 60)
    updated = await store.update_task_due(interaction.user.id, task_id, due_at)
    if not updated:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)
        return

    reminder_note = ""
    if auto_remind:
        reminder = await scheduler.add_reminder(
            interaction.user.id,
            due_in_minutes * 60,
            f"حان موعد المهمة ذات المعرف {task_id}",
            (remind_channel.id if remind_channel else interaction.channel_id),
            metadata={
                "icon": "⏰",
                "prefix": "المهام",
            },
        )
        reminder_note = f"\n🔔 تم إنشاء تذكير (المعرف: {reminder.id})."

    tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
    view = TodoView(store, interaction.user.id, tasks, summary)
    due_text = format_due_relative(due_at) or "قريباً"
    await interaction.response.send_message(
        f"تم تحديد موعد المهمة ليكون {due_text}.{reminder_note}\n\n{view.render_summary()}",
        view=view,
        ephemeral=True,
    )


@todo_group.command(name="complete", description="تعليم مهمة كمكتملة")
async def todo_complete(interaction: discord.Interaction, task_id: str) -> None:
    completed = await store.complete_task(interaction.user.id, task_id)
    if completed:
        tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
        view = TodoView(store, interaction.user.id, tasks, summary)
        await interaction.response.send_message(
            f"تم تعليم المهمة كمكتملة.\n\n{view.render_summary()}",
            view=view,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message("تعذر العثور على المهمة.", ephemeral=True)


@todo_group.command(name="list", description="عرض مهامك الحالية")
async def todo_list(interaction: discord.Interaction) -> None:
    tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
    view = TodoView(store, interaction.user.id, tasks, summary)
    await interaction.response.send_message(view.render_summary(), view=view, ephemeral=True)


@todo_group.command(name="prioritize", description="ترتيب مهامك حسب الأولوية")
async def todo_prioritize(interaction: discord.Interaction) -> None:
    updated = await store.sort_tasks_by_priority(interaction.user.id)
    tasks, summary = await store.get_tasks_with_summary(interaction.user.id)
    view = TodoView(store, interaction.user.id, tasks, summary)
    if updated:
        await interaction.response.send_message(
            "تم ترتيب المهام حسب الأهمية والمواعيد.\n\n" + view.render_summary(),
            view=view,
            ephemeral=True,
        )
    else:
        await interaction.response.send_message(
            view.render_summary(),
            view=view,
            ephemeral=True,
        )


@todo_group.command(name="stats", description="إحصائيات سريعة عن تقدمك")
async def todo_stats(interaction: discord.Interaction) -> None:
    stats = await store.get_task_stats(interaction.user.id)
    message = (
        "**لوحة الإحصاءات:**\n"
        f"• إجمالي المهام: {stats['total']}\n"
        f"• المهام المكتملة: {stats['completed']}\n"
        f"• المهام المتبقية: {stats['pending']}\n"
        f"• المهام المتأخرة: {stats['overdue']}\n"
        f"• المهام عالية الأهمية: {stats['high_priority']}"
    )
    await interaction.response.send_message(message, ephemeral=True)


@todo_group.command(name="advise", description="نصائح ذكية بناءً على مهامك")
async def todo_advise(interaction: discord.Interaction) -> None:
    report = await advisor.build_report(interaction.user.id)
    await interaction.response.send_message(report, ephemeral=True)


@todo_group.command(name="focus", description="بدء جلسة تركيز مع تذكير تلقائي")
@app_commands.describe(
    minutes="مدة جلسة التركيز بالدقائق",
    description="ما الهدف الذي ستركز عليه؟",
    remind_channel="القناة التي سيصلها تنبيه نهاية الجلسة",
)
async def todo_focus(
    interaction: discord.Interaction,
    minutes: app_commands.Range[int, 10, 180] = 25,
    description: Optional[str] = None,
    remind_channel: Optional[discord.TextChannel] = None,
) -> None:
    duration_minutes = minutes or 25
    reminder = await scheduler.add_reminder(
        interaction.user.id,
        duration_minutes * 60,
        description or "انتهت جلسة التركيز! خذ استراحة لطيفة.",
        (remind_channel.id if remind_channel else interaction.channel_id),
        metadata={
            "icon": "🔥",
            "prefix": "تركيز",
        },
    )
    await interaction.response.send_message(
        (
            f"🚀 بدأنا جلسة تركيز لمدة {duration_minutes} دقيقة."
            f" سيتم تنبيهك عند الانتهاء (المعرف: {reminder.id})."
        ),
        ephemeral=True,
    )


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


@remind_group.command(name="smart", description="جدولة تذكير بصياغة طبيعية")
@app_commands.describe(
    phrase="مثال: بعد ساعتين وربع، أو تذكّرني غداً",
    note="ما الرسالة التي تريد سماعها عند التذكير؟",
    channel="القناة التي سيصلها التذكير (اختياري)",
)
async def remind_smart(
    interaction: discord.Interaction,
    phrase: str,
    note: Optional[str] = None,
    channel: Optional[discord.TextChannel] = None,
) -> None:
    seconds = parse_natural_delay(phrase)
    if not seconds:
        await interaction.response.send_message(
            "تعذر فهم الوقت المحدد. حاول استخدام صيغ مثل 'بعد 30 دقيقة' أو 'بعد ساعتين'.",
            ephemeral=True,
        )
        return

    reminder = await scheduler.add_reminder(
        interaction.user.id,
        seconds,
        note or "حان الوقت للتنفيذ!",
        channel.id if channel else interaction.channel_id,
        metadata={
            "icon": "⏰",
            "prefix": "تذكير ذكي",
        },
    )
    await interaction.response.send_message(
        (
            f"تم ضبط التذكير وسيصلك خلال {describe_duration(seconds)} "
            f"(المعرف: {reminder.id})."
        ),
        ephemeral=True,
    )


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
