# Prime Directive (最高指令集)

## 核心行為準則
1. **讀取優先**：在執行任何修改前，必須確認本檔案與 `.antigravity_ignore` 的最新內容。
2. **最高安全性**：嚴禁執行大範圍刪除或可能破壞專案結構的操作。
3. **確認機制**：所有重大變更需先告知使用者欲變更之事項以及可能會發生的解果，並取得使用者同意後才可在實作。
4. **狀態回報**：{每次執行任務前，需確認已讀取此準則。
 檢查是否有與最高優先級規則衝突之處。}
5. **決策機制**：先提出設計方案與資料流。
6. **內斂機制**：經過自我檢查後，才可以產生程式碼或修改內容。

## 使用者偏好設定
- 偏好語言：繁體中文。
- 運作空間：c:\workspace。
- 偏好回答：:{
red_circle: 核心設計（必備）
SRP（Single Responsibility）
Separation of Concerns（SoC）
Layered Architecture
Dependency Inversion（DI）
Abstraction First
Interface-driven design
High cohesion / Low coupling
:orange_circle: 可維護性（Maintainability）
Modularization
Encapsulation
Loose coupling
Configuration over hardcode
Convention over configuration
Orthogonality（正交性）
Design for change
:yellow_circle: 可讀性（Readability）
Intent-revealing naming
Self-documenting code
No magic numbers
Flat structure（avoid deep nesting）
Single level of abstraction
Consistent naming
:green_circle: 可測試性（Testability）
Pure function
Deterministic behavior
Test isolation
Dependency injection
Mockable design
Input/output separation
:blue_circle: 可擴充性（Extensibility）
Open-Closed Principle（OCP）
Plug-in architecture
Strategy pattern
Composition over inheritance
Interface segregation（ISP）
:purple_circle: 除錯與穩定性（Debug / Reliability）
Fail fast
Defensive programming
Assertions
Observability
Structured logging
Traceability
:black_circle: 資料與狀態管理（重要但常被忽略）
Immutable data（盡量）
State isolation
Explicit data flow
Single source of truth
Idempotency
:white_circle: 工程流程（進階）
Separation of build/run
Reproducibility
Versioned config
Experiment tracking
Automation first
}

## 使用者工作區域偏好設定(1.為工作區，2.為資料查找區，3.決策.md)
1. algorithm：存放相關研究工作區，給AI執行、修改
- Optical Flow：光流演算法

2. algorithm_data：存放相關研究資料，給AI查找、了解的
- Optical Flow：光流演算法資料

3. 當每次要執行決策都要產出.md並存放在相對應的algorithm_data下，並且都以時間編輯名稱

## 需要剃除、無用的檔案 (由AI統整、決策，使用者判斷給AI執行刪除) ##
1. 麻煩在你背景運行時所產生的test檔
2. 只用過幾次就沒再用的
3. 只又在那時候重要，但現在已用不到
4. 與主程式執行的關聯性很小
5. 容易讓後續使用者感到困惑

以上決策麻煩統整給我，待我判斷是否刪除