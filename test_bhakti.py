from bhakti_agent import BhaktiAgent, quick_bhakti_recommend, is_devotional
from bhakti_skill import is_bhakti_skill_match, bhakti_score, infer_bhakti_mood, resolve_deity

seed = {
    'title': 'Shreeman Narayan Narayan Hari Hari - Dhun',
    'artists': 'Asif Jeriya, Ami Joshi',
    'album': 'Shreeman Narayan Narayan Hari Hari - Dhun',
    'language': 'hindi',
    'year': '2000',
}

candidates = [
    {'title': 'Kitni Bechain Hoke', 'artists': 'Alka Yagnik, Udit Narayan', 'album': 'Kasoor', 'language': 'hindi', 'year': '2001'},
    {'title': 'Shree Hanuman Chalisa', 'artists': 'Hariharan, Lalit Sen', 'album': 'Shree Hanuman Chalisa', 'language': 'hindi', 'year': '1992'},
    {'title': 'Om Jai Jagdish Hare', 'artists': 'Anup Jalota', 'album': 'Aarti', 'language': 'hindi', 'year': '2005'},
    {'title': 'Tere Liye', 'artists': 'Lata Mangeshkar, Roopkumar Rathod', 'album': 'Veer-Zaara', 'language': 'hindi', 'year': '2004'},
    {'title': 'Hare Krishna Hare Rama', 'artists': 'Traditional', 'album': 'Bhajan', 'language': 'hindi', 'year': '2010'},
    {'title': 'Bekhayali', 'artists': 'Sachet Tandon', 'album': 'Kabir Singh', 'language': 'hindi', 'year': '2019'},
    {'title': 'Shiv Tandav Stotram', 'artists': 'Ravi Shankar', 'album': 'Shiva Stotram', 'language': 'sanskrit', 'year': '2008'},
    {'title': 'Tum Se Hi', 'artists': 'Mohit Chauhan', 'album': 'Jab We Met', 'language': 'hindi', 'year': '2007'},
]

agent = BhaktiAgent(llm_enabled=False)
recommendations = agent.recommend(seed, candidates, top_n=5)

print('=== Bhakti Agent Recommendations for:', seed['title'], '===')
print()
for i, rec in enumerate(recommendations, 1):
    print(str(i) + '. ' + rec['title'] + ' - ' + rec['artists'])
    print('   Score: ' + str(rec['bhakti_score']) + ' | Reason: ' + rec['bhakti_reason'])
    print()

print('=== Devotional Classification ===')
for c in candidates:
    print(c['title'] + ': devotional=' + str(is_devotional(c)))
print(seed['title'] + ': devotional=' + str(is_devotional(seed)))

print()
print('=== Skill Match Test ===')
for c in candidates:
    print(c['title'] + ': bhakti_skill=' + str(is_bhakti_skill_match(c)))

print()
print('=== Mood + Deity Test ===')
for c in [seed] + candidates:
    mood, conf = infer_bhakti_mood(c)
    deity = resolve_deity(c)
    print(c['title'] + ': mood=' + mood + ' (' + str(conf) + ') | deity=' + str(deity))