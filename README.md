# JiSeKa Android

기존 Vercel 웹앱을 WebView로 감싸는 안드로이드 네이티브 앱입니다.

## 기능
- WebView를 통한 Vercel 웹앱 로딩
- 카메라 권한 자동 요청
- `JiSeKaNative` JavaScript 브릿지 인터페이스

## 빌드 방법

이 프로젝트는 로컬에 Android Studio가 없어도 **GitHub Actions**를 통해 자동 빌드됩니다.

1. 이 폴더를 GitHub 레포지토리의 `main` 브랜치에 push
2. GitHub Actions가 자동으로 APK를 빌드
3. Actions 탭 → 최신 워크플로우 → `app-debug` 아티팩트에서 APK 다운로드

## 프로젝트 구조

```
jiseka-android/
├── .github/workflows/android-build.yml   # CI 자동 빌드
├── app/
│   ├── build.gradle                       # 앱 빌드 설정
│   ├── proguard-rules.pro                 # ProGuard 규칙
│   └── src/main/
│       ├── AndroidManifest.xml            # 권한 및 앱 설정
│       ├── java/com/jiseka/app/
│       │   └── MainActivity.kt            # WebView + JS 브릿지
│       └── res/
│           ├── layout/activity_main.xml   # UI 레이아웃
│           └── values/strings.xml         # 문자열 리소스
├── build.gradle                           # 프로젝트 빌드 설정
├── settings.gradle                        # 모듈 설정
├── gradle.properties                      # Gradle 속성
├── gradlew                                # Unix 빌드 스크립트
├── gradlew.bat                            # Windows 빌드 스크립트
└── gradle/wrapper/
    └── gradle-wrapper.properties          # Gradle 래퍼 설정
```
