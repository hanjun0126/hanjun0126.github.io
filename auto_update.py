import os
import subprocess

# 한글 오류 방지 목적
env = os.environ.copy()
env["LC_ALL"] = "C.UTF-8"
env["LANG"] = "C.UTF-8"

# 1. Git 저장소 폴더 설정
GIT_REPO_PATH = r"/Users/hanjunkim/blog"  # Git 저장소 경로

# 2. Git 저장소 폴더로 이동
os.chdir(GIT_REPO_PATH)

# 3. Git 명령어 실행 함수
def run_git_command(command, ignore_error=False):
    # stdout과 stderr을 모두 캡처하여 출력 및 env 전달
    result = subprocess.run(command, capture_output=True, text=True, shell=True, env=env)
    
    # stdout, stderr 둘 다 확인
    output = result.stdout.strip()
    err_output = result.stderr.strip()
    
    if output:
        print(output)
    if err_output:
        print(err_output)
        
    if result.returncode != 0:
        if not ignore_error:
            print(f"⚠️ 명령어 실패 (코드 {result.returncode}): {command}")
        return False
    return True

# 4. push 수행 
print("---- git add ----")
run_git_command("git add -A")

print("---- git commit ----")
# 변경 사항이 없을 경우 커밋은 실패(코드 1)하므로 이 오류를 무시하도록 합니다.
success = run_git_command('git commit -m "Auto: auto update"', ignore_error=True)

if success:
    print("---- git push ----")
    run_git_command("git push")
else:
    print("변경 사항이 없어 푸시를 생략합니다.")
