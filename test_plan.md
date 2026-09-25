# Test Plan

## Functional Test Cases

| Test ID | Test Description | Input | Expected Output | Status |
|---|---|---|---|---|
| T1 | User opens application | Run `streamlit run app.py` | App loads without error | Pass/Fail |
| T2 | User selects liked movies | Select 3 movies | Selected movies appear with rating sliders | Pass/Fail |
| T3 | User changes aspect preferences | Move aspect sliders | Values update on screen | Pass/Fail |
| T4 | User generates recommendations | Click Generate Recommendations | Top 10 movies displayed | Pass/Fail |
| T5 | System excludes selected movies | Select Inception/The Matrix | These movies do not appear in recommendations | Pass/Fail |
| T6 | Evaluation table loads | Open evaluation section | Precision@10 table displayed | Pass/Fail |

## Evaluation Approach

The prototype uses:
- Aspect sentiment matching
- Genre similarity
- Average rating score

The evaluation section tests three sample user profiles:
1. Action and visuals user
2. Drama and acting user
3. Animation and comedy user

The system calculates how many top-10 recommendations match the expected interest categories.
