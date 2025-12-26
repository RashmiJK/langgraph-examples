from langchain_core.messages import HumanMessage

from graph_examples.review_product.research_team import ResearchTeam
from graph_examples.review_product.production_team import ProductionTeam

"""
research_team = ResearchTeam(trace_project_name="ReviewProduct").as_node()

response = research_team.invoke(
    {
        "messages": [
            HumanMessage(content="Hello"),
            HumanMessage(content="How are you?"),
            HumanMessage(
                content="Which one should I buy? eufy E28 Robot vaccum or Dyson Spot Scrub AI robot vaccum? Include youtube as well in your research"
            ),
        ]
    }
)

print(type(response))
print(response)
"""
"""
response = content_writing_agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Which one should I buy? eufy E28 Robot vaccum or Dyson Spot Scrub AI robot vaccum? Include youtube as well in your research"
            ),
            HumanMessage(
                content='{"results":[{"url":"https://www.tomsguide.com/home/eufy-has-just-dropped-a-new-robot-vacuum-that-doubles-as-a-detachable-deep-cleaner","content":"Eufy’s new Robot Vacuum Omni E28 ($999) features a detachable “FlexiOne” deep cleaner for spot cleaning carpets, stairs, and upholstery—an industry first. It offers professional-grade 20,000 Pa Turbo Suction with automatic adjustment of suction and water flow based on mess type. The all-in-one cleaning station self-empties, self-cleans, hot air dries, dispenses detergent, collects wastewater, and self-refills, enabling hands-free maintenance. Additional highlights include DuoSpiral Detangle Brushes for pet hair, CornerRover arm for edge-to-edge cleaning, a HydroJet system for powerful cleaning, and AI-based obstacle avoidance that identifies over 200 objects. It supports customizable modes, no-go zones, and multi-floor cleaning. The reviewer praises its versatility and thorough deep cleaning ability, deeming it a premium option that solves spot-cleaning challenges better than previous models."},{"url":"https://vacuumwars.com/eufy-e28-omni-review-what-you-need-to-know/","content":"Eufy E28 Omni is a 3-in-1 robot vacuum combining vacuuming, roller-style mopping, and a detachable portable spot cleaner. It features strong 20,000 Pa suction, LiDAR navigation with camera-based obstacle avoidance (avoids 23/24 objects in tests), and a multifunction dock that self-empties debris, cleans and dries the mop roller, refills detergent, and manages the spot cleaner’s water tanks. \\n\\nPerformance highlights:  \\n- Excellent surface cleaning on hard floors and carpets; removed 85% of deeply embedded dirt, top 8 of 150+ robots.  \\n- Above-average mopping with a unique roller mop that channels wet spills into a dirty-water tank (stain-removal score 127 vs 93 average).  \\n- Spot cleaner has more suction than Bissell Little Green, effectively cleans stains and wet spills, always charged and ready, with low-maintenance self-flush hose system.  \\n- Navigation efficiency better than average (0.9 m²/min vs 0.7).  \\n- Pet hair pickup matches category average (81%), suitable for moderate shedding but long hair tangles badly, forming “hair cigars”—worst recorded tangle score, a concern for households with very long hair.  \\n\\nCons:  \\n- Suction and airflow slightly below premium robots despite high price.  \\n- Small 300ml dustbin requires frequent returns to dock to empty, shortening run time for large homes.  \\n- Potential durability issues with spot cleaner hose and fittings under heavy daily use.  \\n\\nVerdict:  \\nStrong vacuuming and mopping performance with standout obstacle avoidance and an innovative integrated spot cleaner. Best for mid-size homes and pet owners needing versatile cleaning, but users with very long hair or large floor plans may find limitations. The multifunction dock offers near hands-off maintenance, enhancing convenience. Overall rating 3.86/5, ranking 15th on Vacuum Wars’ Top 20 Robot Vacuums."},{"url":"https://www.youtube.com/watch?v=KxhGvC86jlI","content":"The Yuthi Omni E28 robot vacuum stands out for its HydroJet system, which mimics a wet-dry vacuum by using a rotating roller brush and scraper to effectively clean sticky messes like peanut butter and jam, removing dirty liquid into a tank for superior floor extraction compared to typical robotic mops. Its Dural Spiral Detangling brushes retract and reverse-spin at the base to gather hair centrally, minimizing clogs and keeping rollers clean. The unit features a powerful 20,000 Pascal turbo suction with a dual-hole system for better airflow and anti-clog performance. It integrates advanced AI with up to 200 obstacle recognitions via RGB visual and LIDAR sensors, enabling adaptive cleaning strategies, precise spot cleaning, and a 3D mapping system that completes mapping within three minutes. The clean station includes a large 3L dust bag offering up to 75 days of maintenance-free use, auto water tank flushing (though loud), and hot air drying of mop pads. A notable innovation is the built-in spot cleaner, allowing targeted cleaning on carpets, sofas, or mattresses without separate equipment, enhancing versatility. The robot offers customizable cleaning modes (vacuum, mop, or both), water and suction settings, automatic carpet suction boost, and a unique \\"Smart Track\\" mode that follows the user to clean along their path. Reviewer sentiment is highly positive, emphasizing its exceptional mopping ability—\\"acts like a wet dry vacuum\\"—and dual-purpose design as a good value, though noting smaller tanks compared to standalone wet vacuums and the loud flush of the clean station. Overall, it outperforms typical robo mops in cleaning tough stains and hair management, with advanced navigation and convenient maintenance features."},{"url":"https://www.youtube.com/watch?v=DN29JTFfViw","content":"Eufy E28 robot vacuum mop combo reviewed by Vacuum Wars:\\n\\nPros:\\n- Converts into handheld spot cleaner with water/solution tanks, useful for pet stains; self-cleaning hose feature is easy and convenient.\\n- Strong vacuum performance: picks up various debris well, deep carpet cleaning score 85% (top 8/150 tested).\\n- Excellent obstacle avoidance (23/24 points), aided by front sensors and camera.\\n- Roller-style mop with onboard dirty water tank can handle wet spills better than typical dual spinning pads; above-average stain removal, though may leave more water/streaks.\\n- Advanced multi-functional docking station: auto-empty bin, mop self-wash, dirty water emptying, refill clean water with detergent, hot air drying.\\n- Good navigation and coverage with top-mounted lidar; efficient mapping and above average coverage.\\n- User-friendly app with extensive features: virtual barriers, customization, multi-level maps; highly rated.\\n- Priced ~$300 less than Eufy flagship S1 while retaining many features.\\n\\nCons:\\n- Hair clumping issue (\\"hair cigars\\") behind brush housing after use; some of the worst scores in long hair tests, though not widely reported by others.\\n- Suction and airflow numbers lower than expected for price range.\\n- Small 300 ml onboard dust bin.\\n- Concern about durability of spot cleaner hose/components with heavy daily use; replacements tied to robot vacuum longevity.\\n- Mop may leave streaks due to relatively high water residue.\\n- Battery life slightly below average but with efficient coverage (~1207 sq ft per charge).\\n\\nSummary:\\nA strong vacuum with impressive obstacle avoidance and innovative spot cleaning feature, decent mop ability with potential streaks, but hair clumping on brushes and small bin size are weaknesses. Ranked #15 in Vacuum Wars Top 20 Robot Vacuums. Recommended for pet owners valuing spot cleaner function and excellent navigation, less ideal for homes with long hair shedding."}]}'
            ),
        ]
    }
)

print(type(response))
print(response)
"""
"""
response = audio_synthesis_agent.invoke(
    {
        "messages": [
            HumanMessage(content="Script saved to final_audio_script.txt"),
        ]
    }
)

print(type(response))
print(response)
"""

production_team = ProductionTeam(trace_project_name="ReviewProduct").as_node()

response = production_team.invoke(
    {
        "messages": [
            HumanMessage(content="Hello"),
            HumanMessage(content="How are you?"),
            HumanMessage(
                content='{"results":[{"url":"https://www.pcmag.com/reviews/eufy-omni-e28","content":"**Eufy Omni E28 Robot Vacuum (PCMag Summary)**\\n\\n- **Key Features**: Hybrid robot that vacuums, mops, *and* converts into a portable spot cleaner (similar to a carpet shampoo machine).\\n- **Pros**:\\n    - Unique spot cleaning mode is a serious advantage for tough stains.\\n    - Good overall cleaning on hard floors and carpets.\\n    - Solid navigation and mapping with LiDAR+AI.\\n    - App offers robust controls (custom cleaning zones, scheduling, integration).\\n    - Detachable base for spot cleaning is easy to use and quick to set up.\\n- **Cons**:\\n    - Pricey for mid-tier performance: neither the best vacuum nor mop on the market.\\n    - No automatic dustbin emptying; manual removal needed more often.\\n    - Spot cleaning uses a lot of water: tanks require frequent refills for large messes.\\n    - Tall LiDAR turret has clearance issues under low furniture.\\n    - App and navigation can sometimes struggle with odd-shaped rooms/closed doors.\\n- **Real-World Feedback**:\\n    - Excellent for families with pets/kids, especially messy households.\\n    - If you value low maintenance and never want to manually spot clean, rivals may be more convenient. If you want multi-function and occasional deep cleaning, this is a unique solution.\\n- **Summary**: Not the absolute top performer in either sweeping or mopping, but the spot cleaner mode is unrivaled. Ideal if versatility matters most."},{"url":"https://www.theverge.com/22997597/best-robot-vacuum-cleaner","content":"**The Verge – Best Robot Vacuum Cleaner General Insights**\\n\\n- Dyson’s newest robot vacuums (like the Spot Scrub AI) emphasize advanced AI for navigation, effective avoidance of obstacles, and powerful suction, but are typically *very expensive*. Known for elegant hardware and clever mapping, but sometimes frustrating software and nose-high prices.\\n- Eufy’s robots are generally favored for reliability and value, offering strong suction and sturdy mid-tier performance. Features like hybrid mopping or detachable bases are attractive for busy households, but Eufy does not compete with Dyson’s navigation/mapping sophistication.\\n- The *trade-off*: Eufy can be an excellent value—especially for spot cleaning versatility—but if you want the “smartest” robot (especially multi-room mapping, complex schedules, and boundary recognition), Dyson’s Spot Scrub AI is probably the leader. Both brands have minor quirks like getting stuck under furniture or leaving streaks after mopping.\\n- **Neither is perfect**: Dyson is for tech-lovers who want maximum autonomy and polish; Eufy targets those who want hybrid use and a reasonable price."}]}'
            ),
        ]
    }
)

print(type(response))
print(response)
