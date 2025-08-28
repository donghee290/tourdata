from reporting import indicator_calculator, report_generator

def main():
    print("=== [1] 지표 산출 시작 ===")
    indicator_calculator.main()

    print("\n=== [2] 리포트 생성 시작 ===")
    report_generator.main()

    print("\n=== 전체 파이프라인 완료 ===")

if __name__ == "__main__":
    main()