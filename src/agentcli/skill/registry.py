from __future__ import annotations

import json
import re
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path
    content: str
    source: str = "project"
    version: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def body(self) -> str:
        return _strip_frontmatter(self.content).strip()


class SkillMatcher:
    """Rank enabled skills against a user request using lightweight lexical signals."""

    def __init__(self, skills: Iterable[Skill]):
        self.skills = tuple(skill for skill in skills if skill.enabled)

    def match(self, query: str, *, top_k: int = 5) -> list[Skill]:
        if top_k <= 0 or not query.strip():
            return []
        query_terms = _match_terms(query)
        ranked: list[tuple[int, str, Skill]] = []
        for skill in self.skills:
            score = self._score(query, query_terms, skill)
            if score > 0:
                ranked.append((score, skill.name, skill))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _score, _name, skill in ranked[:top_k]]

    @staticmethod
    def _score(query: str, query_terms: set[str], skill: Skill) -> int:
        score = 0
        if _explicit_skill_name(query, skill.name):
            score += 10_000 + len(_compact_match_text(skill.name))

        name_terms = _match_terms(skill.name)
        tag_terms = _match_terms(" ".join(skill.tags))
        description_terms = _match_terms(skill.description)
        score += 12 * len(query_terms & name_terms)
        score += 6 * len(query_terms & tag_terms)
        score += 2 * len(query_terms & description_terms)
        return score


class SkillContextBuffer:
    def __init__(self, limit: int = 3):
        self.limit = limit
        self._items: OrderedDict[str, str] = OrderedDict()

    def push(self, name: str | None, body: str | None) -> None:
        if not name or not body:
            return
        if name in self._items:
            del self._items[name]
        self._items[name] = body
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def drain(self) -> str:
        if not self._items:
            return ""
        chunks = [
            f"## Loaded Skill: {name}\n{body.strip()}"
            for name, body in self._items.items()
            if body.strip()
        ]
        self._items.clear()
        return "\n\n".join(chunks)

    def clear(self) -> None:
        self._items.clear()

    def is_empty(self) -> bool:
        return not self._items

    def size(self) -> int:
        return len(self._items)


class SkillStateStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".agentcli" / "skills.json").expanduser()

    def disabled(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        values = data.get("disabled") if isinstance(data, dict) else None
        if not isinstance(values, list):
            return set()
        return {str(item) for item in values if str(item).strip()}

    def disable(self, name: str) -> None:
        values = self.disabled()
        values.add(name)
        self._write(values)

    def enable(self, name: str) -> None:
        values = self.disabled()
        values.discard(name)
        self._write(values)

    def _write(self, disabled: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class SkillRegistry:
    """Load SKILL.md files from built-in, user, and project locations."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        builtin_root: str | Path | None = None,
        user_root: str | Path | None = None,
        state_store: SkillStateStore | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        package_root = Path(__file__).resolve().parents[1]
        self.builtin_root = Path(builtin_root or package_root / "builtin_skills")
        self.user_root = Path(user_root or Path.home() / ".agentcli" / "skills")
        self.project_skill_root = self.project_root / ".agentcli" / "skills"
        self.state_store = state_store or SkillStateStore()
        self._skills: dict[str, Skill] | None = None

    def reload(self) -> None:
        self._skills = None

    def list(self) -> list[Skill]:
        return self.enabled_skills()

    def all_skills(self) -> list[Skill]:
        skills = self._load_all()
        return [skills[name] for name in sorted(skills)]

    def enabled_skills(self) -> list[Skill]:
        return [skill for skill in self.all_skills() if skill.enabled]

    def load(self, name: str, *, include_disabled: bool = False) -> Skill | None:
        skill = self._load_all().get(name)
        if not skill:
            return None
        if not include_disabled and not skill.enabled:
            return None
        return skill

    def enable(self, name: str) -> bool:
        if not self.load(name, include_disabled=True):
            return False
        self.state_store.enable(name)
        self.reload()
        return True

    def disable(self, name: str) -> bool:
        if not self.load(name, include_disabled=True):
            return False
        self.state_store.disable(name)
        self.reload()
        return True

    def match(self, query: str, *, top_k: int = 5) -> list[Skill]:
        return SkillMatcher(self.enabled_skills()).match(query, top_k=top_k)

    def create(
        self,
        name: str,
        *,
        description: str,
        body: str,
        scope: str = "project",
        version: str = "1.0.0",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> Skill:
        slug = _validate_skill_slug(name)
        description = description.strip()
        body = body.strip()
        if not description:
            raise ValueError("skill description must not be empty")
        if not body:
            raise ValueError("skill body must not be empty")

        path = self._skill_path(slug, scope)
        if path.exists() and not overwrite:
            raise FileExistsError(f'skill "{slug}" already exists in {scope} scope')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _render_skill_file(
                name=slug,
                description=description,
                body=body,
                version=version,
                tags=tags or [],
            ),
            encoding="utf-8",
        )
        self.reload()
        skill = self._load_skill_file(path, scope, self.state_store.disabled())
        if not skill:  # pragma: no cover - the file was just written successfully
            raise OSError(f"failed to load newly written skill: {path}")
        return skill

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        body: str | None = None,
        scope: str = "project",
        version: str | None = None,
        tags: list[str] | None = None,
    ) -> Skill:
        slug = _validate_skill_slug(name)
        path = self._skill_path(slug, scope)
        if not path.is_file():
            raise FileNotFoundError(f'skill "{slug}" does not exist in {scope} scope')
        existing = self._load_skill_file(path, scope, self.state_store.disabled())
        if not existing:
            raise ValueError(f'skill "{slug}" could not be parsed')
        return self.create(
            slug,
            description=description if description is not None else existing.description,
            body=body if body is not None else existing.body,
            scope=scope,
            version=version if version is not None else existing.version,
            tags=tags if tags is not None else existing.tags,
            overwrite=True,
        )

    def index_text(self, max_chars: int = 4000, max_skills: int = 20) -> str:
        skills = self.enabled_skills()[:max_skills]
        if not skills:
            return ""
        lines = [
            "Available skills:",
            "Load a skill with load_skill(name) when its description matches the task.",
        ]
        for skill in skills:
            description = " ".join(skill.description.split())
            if len(description) > 500:
                description = description[:497] + "..."
            lines.append(f"- {skill.name}: {description}")
        text = "\n".join(lines)
        return text[:max_chars]

    def _scope_root(self, scope: str) -> Path:
        if scope == "project":
            root = self.project_skill_root.resolve()
            try:
                root.relative_to(self.project_root)
            except ValueError as exc:
                raise ValueError("project skill root must stay inside the project") from exc
            return root
        if scope == "user":
            return self.user_root.expanduser().resolve()
        raise ValueError('skill scope must be "project" or "user"')

    def _skill_path(self, name: str, scope: str) -> Path:
        root = self._scope_root(scope)
        path = root / name / "SKILL.md"
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("skill path must stay inside its scope root") from exc
        return path

    def _load_all(self) -> dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        disabled = self.state_store.disabled()
        skills: dict[str, Skill] = {}
        for source, root in [
            ("builtin", self.builtin_root),
            ("user", self.user_root),
            ("project", self.project_skill_root),
        ]:
            if not root.exists():
                continue
            for skill_file in sorted(root.glob("*/SKILL.md")):
                skill = self._load_skill_file(skill_file, source, disabled)
                if skill:
                    skills[skill.name] = skill
        self._skills = skills
        return skills

    def _load_skill_file(self, path: Path, source: str, disabled: set[str]) -> Skill | None:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        metadata = _parse_frontmatter(content)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description") or ""
        tags = _parse_tags(metadata.get("tags", ""))
        return Skill(
            name=name,
            description=description,
            version=metadata.get("version") or "",
            tags=tags,
            source=source,
            path=path,
            content=content,
            enabled=name not in disabled,
        )


def _parse_frontmatter(content: str) -> dict[str, str]:
    if not content.startswith("---"):
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*", content, re.S)
    if not match:
        return {}
    lines = match.group(1).splitlines()
    metadata: dict[str, str] = {}
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        if ":" not in raw_line:
            index += 1
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "|":
            index += 1
            block: list[str] = []
            while index < len(lines) and (lines[index].startswith(" ") or not lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            metadata[key] = " ".join(part for part in block if part)
            continue
        metadata[key] = value.strip().strip('"').strip("'")
        index += 1
    return metadata


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    return re.sub(r"^---\s*\n.*?\n---\s*", "", content, count=1, flags=re.S)


def _parse_tags(raw: str) -> list[str]:
    value = raw.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]


_SKILL_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_ASCII_TERM = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_MATCH_STOPWORDS = {"and", "for", "from", "the", "this", "use", "with"}


def _validate_skill_slug(name: str) -> str:
    if not isinstance(name, str) or not _SKILL_SLUG.fullmatch(name):
        raise ValueError(
            "skill name must be a lowercase slug using letters, numbers, hyphens, or underscores"
        )
    return name


def _render_skill_file(
    *,
    name: str,
    description: str,
    body: str,
    version: str,
    tags: list[str],
) -> str:
    clean_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    lines = ["---", f"name: {name}", "description: |"]
    lines.extend(f"  {line}" for line in description.splitlines())
    lines.extend(
        [
            f"version: {json.dumps(str(version), ensure_ascii=False)}",
            f"tags: {json.dumps(clean_tags, ensure_ascii=False)}",
            "---",
            "",
            body.rstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


def _compact_match_text(value: str) -> str:
    return _normalize_match_text(value).replace(" ", "")


def _explicit_skill_name(query: str, name: str) -> bool:
    normalized_query = _normalize_match_text(query)
    normalized_name = _normalize_match_text(name)
    if not normalized_name:
        return False
    tokens = normalized_name.split()
    pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(token) for token in tokens)
    pattern += r"(?![a-z0-9])"
    return re.search(pattern, normalized_query) is not None


def _match_terms(value: str) -> set[str]:
    normalized = _normalize_match_text(value)
    terms = {
        term
        for term in _ASCII_TERM.findall(normalized)
        if len(term) >= 2 and term not in _MATCH_STOPWORDS
    }
    for run in _CJK_RUN.findall(normalized):
        if len(run) <= 4:
            terms.add(run)
        for size in (2, 3):
            terms.update(run[index : index + size] for index in range(len(run) - size + 1))
    return terms
