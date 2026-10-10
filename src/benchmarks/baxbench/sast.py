import json
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any

_SNYK_CLI_PATH = "/tmp/snyk-cli"


@dataclass
class SASTReport:
    # The path to file where the vuln was found
    # Uses `str` instead of `pathlib.Path` to make it JSON-serializable.
    file_path: str
    cwe_id: int
    message: str
    line_num: int
    # A vendor-specific rule id, which may be different from the `cwe_id`
    rule_id: str


def run_snyk_code(results_dir: pathlib.Path, force: bool) -> list[SASTReport]:
    if not pathlib.Path(_SNYK_CLI_PATH).is_file():
        # Download the Snyk CLI and authenticate if necessary
        print(f"downloading the Snyk cli to: {_SNYK_CLI_PATH}")
        subprocess.check_call(
            [
                "curl",
                "--compressed",
                "https://downloads.snyk.io/cli/stable/snyk-macos",
                "-o",
                _SNYK_CLI_PATH,
            ]
        )
        subprocess.check_call(["chmod", "+x", _SNYK_CLI_PATH])
        print("authenticating with Snyk")
        subprocess.check_call([_SNYK_CLI_PATH, "auth"])
    json_out_file = results_dir / "snyk_code_out.json"

    # Run the CLI, if necessary.
    if pathlib.Path(json_out_file).exists() and not force:
        print(
            f"Reusing results from {json_out_file}. Please remove it manually if you want to rerun the scanning"
        )
    else:
        args = [
            _SNYK_CLI_PATH,
            "code",
            "test",
            f"--json-file-output={str(json_out_file)}",
            str(results_dir),
        ]
        print("running Snyk Code: ", args)
        subprocess.call(args)

    # Read the results.
    with open(json_out_file, "r") as f:
        sarif_json = json.load(f)

    # Map rule ids to CWE numbers.
    def parse_rule(d: dict[Any, Any]) -> tuple[str, int]:
        cwe_id = int(d["properties"]["cwe"][0][len("CWE-") :])
        return (d["id"], cwe_id)

    rule_id_to_cwe: dict[str, int] = dict(
        (parse_rule(r) for r in sarif_json["runs"][0]["tool"]["driver"]["rules"])
    )

    # Parse the reports.
    def parse_report(d: dict[Any, Any]) -> SASTReport:
        loc = d["locations"][0]["physicalLocation"]
        return SASTReport(
            file_path=str(results_dir / loc["artifactLocation"]["uri"]),
            cwe_id=rule_id_to_cwe[d["ruleId"]],
            message=d["message"]["text"],
            line_num=int(loc["region"]["startLine"]),
            rule_id=d["ruleId"],
        )

    return [parse_report(d) for d in sarif_json["runs"][0]["results"]]
