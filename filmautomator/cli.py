"""Owner command-line interface (spec section 18).

The owner interface is deliberately a CLI in this milestone: it is the
smallest thing that exposes every control the spec asks for (START, PAUSE,
RESUME, STOP, RETRY, APPROVE, REJECT, plus natural-language direction to the
Director) without building a UI before the engine works.

A web control panel is a thin layer over the same :class:`Producer` API.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .blender.session import BlenderSession
from .config import load_config
from .core.project import ProjectDB
from .producer import DependencyMissing, Producer, ProductionRequest

BANNER = "FilmAutomator — autonomous, zero-budget AI filmmaking"


def _setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Blender's own chatter and urllib noise drown out our own logs.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _progress(message: str, agent: str = "", **_kwargs: object) -> None:
    prefix = f"[{agent}] " if agent else ""
    print(f"  {prefix}{message}", flush=True)


def _registry(config) -> ProjectDB:
    return ProjectDB(Path(config.workspace) / "registry.db")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report on every external dependency and the model gateway."""
    config = load_config()
    print(BANNER)
    print()
    print("External tools")
    print("-" * 60)
    print(BlenderSession.diagnose())
    print()
    from .post.ffmpeg import VideoEncoder
    print(VideoEncoder().diagnose())
    print()

    print("Model Gateway")
    print("-" * 60)
    producer = Producer(config)
    print(producer.gateway_report())
    print()

    missing: list[str] = []
    if config.blender.executable is None:
        missing.append("blender")
    if not producer.encoder.available:
        missing.append("ffmpeg")
    if missing:
        print("Not ready: missing " + ", ".join(missing))
        print("Run:  brew install --cask blender ffmpeg")
        return 1
    print("Ready to produce.")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Plan a sequence and print the shot list, without rendering anything."""
    config = load_config()
    producer = Producer(config)
    try:
        producer.preflight(require_ffmpeg=False)
    except DependencyMissing as exc:
        print(str(exc), file=sys.stderr)
        return 1

    from .agents import AgentContext, BlenderAgent, Director, VisionAgent
    from .core.workspace import Workspace

    project_id, workspace, db = producer.start(
        ProductionRequest(objective=args.objective, duration_s=args.duration),
        require_ffmpeg=False,
    )
    try:
        context = AgentContext(
            gateway=producer.gateway, config=config, project_id=project_id,
            db=db, workspace=workspace, session=producer.session,
        )
        director = Director(
            context,
            BlenderAgent(context, producer.session, config.render),
            VisionAgent(context),
            producer.encoder,
        )
        plan = director.plan_shots(args.objective, args.duration, style=args.style or "")
        print()
        print(f"Title:   {plan.title}")
        print(f"Logline: {plan.logline}")
        print()
        total = 0.0
        for spec in plan.shots:
            total += spec.duration_s
            print(f"{spec.shot_id}  {spec.duration_s:5.1f}s  "
                  f"{spec.camera.shot_size:<16} {spec.camera.angle:<12} "
                  f"{spec.camera.lens_mm:.0f}mm  {spec.lighting.mood}")
            print(f"    {spec.description}")
            for criterion in spec.quality_criteria:
                print(f"      criteria: {criterion}")
        print()
        print(f"Total duration: {total:.1f}s")
        for note in plan.notes:
            print(f"Note: {note}")
        return 0
    finally:
        producer.stop()


def cmd_make(args: argparse.Namespace) -> int:
    """The full pipeline: objective in, movie out."""
    config = load_config()
    if args.width:
        config.render.final_width = args.width
    if args.height:
        config.render.final_height = args.height
    if args.engine:
        config.render.final_engine = args.engine
    if args.rounds:
        config.limits.max_shot_revision_rounds = args.rounds

    producer = Producer(config, progress=_progress)
    print(BANNER)
    print()

    request = ProductionRequest(
        objective=args.objective,
        duration_s=args.duration,
        style=args.style or "",
        title=args.title or "",
        extra_context=args.context or "",
        offline=args.offline,
        plan_file=args.plan or "",
    )

    try:
        result = producer.run(request)
    except DependencyMissing as exc:
        print()
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        producer.stop()

    print()
    print("=" * 60)
    print(result.summary())
    if result.qa:
        print()
        print(result.qa.render())
    return 0 if result.success else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show progress for a project."""
    config = load_config()
    with _registry(config) as db:
        projects = db.list_projects()
        if not projects:
            print("No projects yet. Run:  filmautomator make \"...\"")
            return 0

        target = args.project_id or projects[0]["project_id"]
        project = db.get_project(target)
        if project is None:
            print(f"No such project: {target}", file=sys.stderr)
            return 1

        print(f"Project:  {project['name']}  ({target})")
        print(f"Objective: {project['objective']}")
        print(f"Created:  {project['created_at']}")
        print()

        progress = db.progress(target)
        print(f"Shots:    {progress['shots_approved']}/{progress['shots_total']} "
              f"approved ({progress['shots_percent']}%)")
        print(f"Tasks:    {progress['tasks_total']}")
        for status, count in sorted(progress["tasks_by_status"].items()):
            print(f"            {status:<12} {count}")
        print(f"Assets:   {progress['assets']}")
        print(f"Characters: {progress['characters']}")
        print()

        shots = db.list_shots(target)
        if shots:
            print("Shots")
            print("-" * 60)
            for shot in shots:
                versions = db.list_shot_versions(target, shot["shot_id"])
                print(f"  {shot['shot_id']}  {shot['duration_s']:5.1f}s  "
                      f"{shot['status']:<12} {len(versions)} version(s)")
                for version in versions:
                    mark = "approved" if version.approved else "        "
                    qa_score = version.qa.get("score")
                    score_text = f"score {qa_score:.2f}" if isinstance(qa_score, (int, float)) else ""
                    print(f"      v{version.version_no:03d} {mark} {score_text}")

        pending = db.pending_decisions(target)
        if pending:
            print()
            print(f"{len(pending)} decision(s) awaiting you")
            print("-" * 60)
            for decision in pending:
                print(f"  [{decision['decision_id']}] {decision['question']}")
                for option in decision["options"]:
                    print(f"        - {option}")
                print(f"      answer with: filmautomator answer "
                      f"{decision['decision_id']} \"your choice\"")
        return 0


def cmd_decisions(args: argparse.Namespace) -> int:
    config = load_config()
    with _registry(config) as db:
        projects = db.list_projects()
        target = args.project_id or (projects[0]["project_id"] if projects else "")
        if not target:
            print("No projects yet.")
            return 0
        pending = db.pending_decisions(target)
        if not pending:
            print("Nothing awaiting your decision.")
            return 0
        for decision in pending:
            print(f"[{decision['decision_id']}] ({decision['urgency']}) "
                  f"{decision['question']}")
            for option in decision["options"]:
                print(f"    - {option}")
        return 0


def cmd_answer(args: argparse.Namespace) -> int:
    config = load_config()
    with _registry(config) as db:
        decision = db.get_decision(args.decision_id)
        if decision is None:
            print(f"No such decision: {args.decision_id}", file=sys.stderr)
            return 1
        db.resolve_decision(args.decision_id, args.answer)
        db.log_event(decision["project_id"], "decision_resolved", args.answer,
                     {"decision_id": args.decision_id})
        print(f"Recorded: {args.answer}")
        return 0


def cmd_events(args: argparse.Namespace) -> int:
    config = load_config()
    with _registry(config) as db:
        projects = db.list_projects()
        target = args.project_id or (projects[0]["project_id"] if projects else "")
        if not target:
            print("No projects yet.")
            return 0
        for event in reversed(db.list_events(target, limit=args.limit)):
            print(f"{event['created_at']}  {event['kind']:<22} {event['message']}")
        return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="filmautomator",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  filmautomator doctor\n"
            "  filmautomator make \"a lone figure on a rain-soaked street at night\" "
            "--duration 10\n"
            "  filmautomator status\n"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="errors only")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check dependencies and model availability").set_defaults(
        func=cmd_doctor
    )

    plan = sub.add_parser("plan", help="plan a sequence without rendering")
    plan.add_argument("objective", help="what the scene should be")
    plan.add_argument("--duration", type=float, default=10.0,
                      help="target total duration in seconds")
    plan.add_argument("--style", default="", help="visual style guidance")
    plan.set_defaults(func=cmd_plan)

    make = sub.add_parser("make", help="produce a movie from an objective")
    make.add_argument("objective", help="what the movie should be")
    make.add_argument("--duration", type=float, default=10.0,
                      help="target total duration in seconds")
    make.add_argument("--style", default="", help="visual style guidance")
    make.add_argument("--title", default="", help="project title")
    make.add_argument("--context", default="", help="extra direction for the Director")
    make.add_argument("--engine", default="",
                      help="final render engine: CYCLES | BLENDER_EEVEE_NEXT | "
                           "BLENDER_WORKBENCH")
    make.add_argument("--width", type=int, default=0, help="final render width")
    make.add_argument("--height", type=int, default=0, help="final render height")
    make.add_argument("--rounds", type=int, default=0,
                      help="max revision rounds per shot (default from config)")
    make.add_argument("--offline", action="store_true",
                      help="skip the model and use the deterministic fallback plan")
    make.add_argument("--plan", default="",
                      help="JSON file with your own shot list, used instead of "
                           "asking the Director to plan")
    make.set_defaults(func=cmd_make)

    status = sub.add_parser("status", help="show project progress")
    status.add_argument("project_id", nargs="?", default="")
    status.set_defaults(func=cmd_status)

    decisions = sub.add_parser("decisions", help="list decisions awaiting the owner")
    decisions.add_argument("project_id", nargs="?", default="")
    decisions.set_defaults(func=cmd_decisions)

    answer = sub.add_parser("answer", help="answer a pending decision")
    answer.add_argument("decision_id")
    answer.add_argument("answer")
    answer.set_defaults(func=cmd_answer)

    events = sub.add_parser("events", help="show the production event log")
    events.add_argument("project_id", nargs="?", default="")
    events.add_argument("--limit", type=int, default=40)
    events.set_defaults(func=cmd_events)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, args.quiet)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
