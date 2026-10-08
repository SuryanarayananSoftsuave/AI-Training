# Error before/after transcript

**Question:** What does the leave policy say, as of 2000-01-01?

## Before
1. `search_handbook({'as_of': '2000-01-01', 'query': 'leave policy'})` -> Error calling tool 'search_handbook': not found

**Final answer:** I'm sorry, but I wasn't able to retrieve the leave policy from the HR handbook — the search returned a "not found" error. This could mean the policy isn't available for that date, or the handbook isn't accessible.

If you can share the policy text or clarify the topic, I'd be happy to help interpret it.

## After
1. `search_handbook({'as_of': '2000-01-01', 'query': 'leave policy'})` -> Error calling tool 'search_handbook': no policy version effective 2000-01-01: earliest indexed is 2026-09-02

**Final answer:** There is no leave policy version effective on 2000-01-01. The earliest version indexed in the handbook is **2026-09-02**, so no policy text from that date exists to retrieve.

If you'd like, I can pull the leave policy from the earliest available version (2026-09-02) or the currently indexed version instead.
