# Test Plan

| Test ID | Test Description | Expected Result | Status |
|---|---|---|---|
| T1 | Open the Streamlit app | App loads without errors | Pass/Fail |
| T2 | Dataset download | MovieLens data is downloaded and dataset counts are displayed | Pass/Fail |
| T3 | Select liked movies | Selected movies are accepted as user input | Pass/Fail |
| T4 | Adjust aspect sliders | Aspect preferences update correctly | Pass/Fail |
| T5 | Generate recommendations | A ranked list of movies is displayed | Pass/Fail |
| T6 | Recommendation explanation | Each recommendation shows title, genres, average rating and matching aspects | Pass/Fail |
| T7 | Run quick evaluation | Offline evaluation produces Hit Rate@10 and baseline values | Pass/Fail |
| T8 | Invalid/no movie selection | System still produces recommendations based on aspect preferences | Pass/Fail |
