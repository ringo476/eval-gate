"""
The document corpus the rag_fusion adapter retrieves from. Deliberately
small (24 short documents, 8 topics) so it's fully readable and every
golden-set case's expected relevant_doc_ids can be hand-verified against
it -- the point of a golden set is that a human can audit it, not that
it's large.
"""

CORPUS: dict[str, str] = {
    # --- economic impact ---
    "doc1": "Climate change is projected to reduce global GDP by several percentage "
            "points by 2100 under high-emissions scenarios, driven by lower "
            "agricultural yields, heat-related productivity losses, and damage to "
            "coastal infrastructure from sea-level rise.",
    "doc2": "Insurance markets in flood- and wildfire-prone regions have seen "
            "premiums rise sharply, and some insurers have withdrawn coverage "
            "entirely from the highest-risk areas, shifting costs onto homeowners "
            "and public disaster-relief programs.",
    "doc3": "The transition to renewable energy has created new jobs in solar and "
            "wind manufacturing and installation, but has also displaced workers "
            "in coal mining and fossil-fuel power generation, requiring targeted "
            "retraining programs.",

    # --- public health ---
    "doc4": "Rising average temperatures are expanding the geographic range of "
            "disease-carrying mosquitoes, increasing the risk of malaria and "
            "dengue fever in regions that were previously too cool for them.",
    "doc5": "Heatwaves linked to climate change cause thousands of excess deaths "
            "annually, with the elderly, outdoor workers, and people without "
            "access to air conditioning at the highest risk.",
    "doc6": "Wildfire smoke, made more frequent and intense by hotter, drier "
            "conditions, is linked to increased rates of asthma, cardiovascular "
            "disease, and premature death in affected populations.",

    # --- policy ---
    "doc7": "The Paris Agreement commits signatory countries to limiting global "
            "warming to well below 2 degrees Celsius above pre-industrial levels, "
            "with an aspirational target of 1.5 degrees.",
    "doc8": "Carbon pricing mechanisms, including carbon taxes and cap-and-trade "
            "systems, are designed to make emitting greenhouse gases costlier, "
            "incentivizing businesses to reduce their carbon footprint.",
    "doc9": "Many national climate policies now include adaptation funding "
            "alongside emissions-reduction targets, recognizing that some degree "
            "of climate change is already locked in regardless of future action.",

    # --- biodiversity ---
    "doc10": "Coral reefs are experiencing mass bleaching events as ocean "
             "temperatures rise, with some reef systems losing more than half "
             "their coral cover within a single decade.",
    "doc11": "Shifting temperature and precipitation patterns are pushing many "
             "species to migrate toward the poles or to higher elevations, "
             "disrupting existing ecosystems and food webs.",
    "doc12": "Ocean acidification, caused by seawater absorbing excess "
             "atmospheric carbon dioxide, makes it harder for shellfish and "
             "other marine organisms to build their calcium carbonate shells.",

    # --- technology & mitigation ---
    "doc13": "Utility-scale battery storage has become significantly cheaper over "
             "the past decade, making it more practical to pair with solar and "
             "wind power to supply electricity when the sun isn't shining or the "
             "wind isn't blowing.",
    "doc14": "Direct air capture technology extracts carbon dioxide directly from "
             "the atmosphere, but currently remains expensive and energy-intensive "
             "compared to reducing emissions at the source.",
    "doc15": "Electrification of transportation, including electric cars, buses, "
             "and increasingly trucks, is a major lever for reducing emissions in "
             "the transport sector, provided the electricity grid itself is "
             "decarbonized.",

    # --- climate science ---
    "doc16": "Climate models consistently project that continued greenhouse gas "
             "emissions will lead to further warming, more frequent extreme "
             "weather events, and continued sea-level rise over the coming "
             "century.",
    "doc17": "The greenhouse effect occurs when gases like carbon dioxide and "
             "methane trap heat in the atmosphere that would otherwise radiate "
             "back into space, warming the planet's surface.",
    "doc18": "Ice cores drilled from Antarctica and Greenland provide a record of "
             "atmospheric composition going back hundreds of thousands of years, "
             "showing that current carbon dioxide levels are unprecedented in that "
             "timeframe.",

    # --- agriculture ---
    "doc19": "Shifting rainfall patterns and more frequent droughts are reducing "
             "crop yields in several major agricultural regions, threatening food "
             "security for populations that depend on local harvests.",
    "doc20": "Some farmers are adapting to changing conditions by switching to "
             "more drought-resistant crop varieties or adjusting planting "
             "schedules to match shifted growing seasons.",
    "doc21": "Livestock farming is both affected by and a contributor to climate "
             "change, as heat stress reduces animal productivity while methane "
             "from digestion is a significant greenhouse gas source.",

    # --- migration & displacement ---
    "doc22": "Rising sea levels are expected to displace tens of millions of "
             "people in low-lying coastal areas and island nations over the "
             "coming decades, raising difficult questions about resettlement and "
             "statelessness.",
    "doc23": "Prolonged droughts have been linked to internal displacement and "
             "migration in agricultural communities, as failed harvests push "
             "people toward cities or across borders in search of work.",
    "doc24": "Climate-related displacement is not yet formally recognized under "
             "international refugee law, leaving many people displaced by "
             "environmental factors without a clear legal status.",
}
