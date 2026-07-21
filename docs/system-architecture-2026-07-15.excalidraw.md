---
excalidraw-plugin: parsed
tags: [excalidraw]
---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠== You can decompress Drawing data with the command palette: 'Decompress current Excalidraw file'. For more info check in plugin settings under 'Saving'

# Excalidraw Data

## Text Elements
%%
## Drawing
```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "https://github.com/zsviczian/obsidian-excalidraw-plugin",
  "elements": [
    {
      "type": "text",
      "id": "title",
      "x": 260,
      "y": 10,
      "text": "AMD ACT II Hackathon \u2014 Full System Architecture",
      "fontSize": 28,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "originalText": "AMD ACT II Hackathon \u2014 Full System Architecture",
      "autoResize": true
    },
    {
      "type": "text",
      "id": "subtitle",
      "x": 340,
      "y": 48,
      "text": "Task Input \u2192 Classification \u2192 Routing \u2192 Solving \u2192 QC \u2192 Aggregation \u2192 Answer Output",
      "fontSize": 16,
      "fontFamily": 1,
      "strokeColor": "#757575",
      "originalText": "Task Input \u2192 Classification \u2192 Routing \u2192 Solving \u2192 QC \u2192 Aggregation \u2192 Answer Output",
      "autoResize": true
    },
    {
      "type": "rectangle",
      "id": "bg_input",
      "x": 30,
      "y": 90,
      "width": 180,
      "height": 80,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#a5d8ff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_bg_input",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_bg_input",
      "x": 35,
      "y": 100,
      "width": 170,
      "height": 60,
      "text": "INPUT\n/input/tasks.json\nor stdin (TASK_COUNT)",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "bg_input",
      "originalText": "INPUT\n/input/tasks.json\nor stdin (TASK_COUNT)",
      "autoResize": true
    },
    {
      "type": "rectangle",
      "id": "prefilter",
      "x": 280,
      "y": 90,
      "width": 180,
      "height": 80,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#a5d8ff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_prefilter",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_prefilter",
      "x": 285,
      "y": 100,
      "width": 170,
      "height": 60,
      "text": "PRE-FILTER (T0/T1)\nGreetings \u2192 bypass\nCode detection \u2192 route\nArithmetic \u2192 bypass",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "prefilter",
      "originalText": "PRE-FILTER (T0/T1)\nGreetings \u2192 bypass\nCode detection \u2192 route\nArithmetic \u2192 bypass",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_input_to_prefilter",
      "x": 210,
      "y": 130,
      "width": 70,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          70,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "bg_input",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "prefilter",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "diamond",
      "id": "bypass_diamond",
      "x": 530,
      "y": 105,
      "width": 100,
      "height": 70,
      "backgroundColor": "#b2f2bb",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_bypass_diamond",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_bypass_diamond",
      "x": 540,
      "y": 120,
      "width": 80,
      "height": 40,
      "text": "Bypass?\ndirect_answer?",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "bypass_diamond",
      "originalText": "Bypass?\ndirect_answer?",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_prefilter_to_bypass",
      "x": 460,
      "y": 130,
      "width": 70,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          70,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "prefilter",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "bypass_diamond",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "bypass_output",
      "x": 700,
      "y": 110,
      "width": 130,
      "height": 50,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#b2f2bb",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_bypass_output",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_bypass_output",
      "x": 705,
      "y": 120,
      "width": 120,
      "height": 30,
      "text": "DIRECT ANSWER\n(no LLM cost)",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "bypass_output",
      "originalText": "DIRECT ANSWER\n(no LLM cost)",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_bypass_to_output",
      "x": 630,
      "y": 140,
      "width": 70,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          70,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "bypass_diamond",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "bypass_output",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "classifier",
      "x": 280,
      "y": 220,
      "width": 180,
      "height": 90,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#a5d8ff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_classifier",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_classifier",
      "x": 285,
      "y": 230,
      "width": 170,
      "height": 70,
      "text": "CATEGORY CLASSIFIER\n8-way deterministic\n+ secondary classifiers\n(code, reasoning, factual, summ)",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "classifier",
      "originalText": "CATEGORY CLASSIFIER\n8-way deterministic\n+ secondary classifiers\n(code, reasoning, factual, summ)",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_prefilter_to_classifier",
      "x": 370,
      "y": 170,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "prefilter",
        "fixedPoint": [
          0.5,
          1
        ]
      },
      "endBinding": {
        "elementId": "classifier",
        "fixedPoint": [
          0.5,
          0
        ]
      },
      "strokeStyle": "dashed"
    },
    {
      "type": "rectangle",
      "id": "complexity",
      "x": 280,
      "y": 360,
      "width": 180,
      "height": 60,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#a5d8ff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_complexity",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_complexity",
      "x": 285,
      "y": 368,
      "width": 170,
      "height": 44,
      "text": "COMPLEXITY SCORER\nMLM per-category\n0.0 (simple) \u2192 1.0 (hard)",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "complexity",
      "originalText": "COMPLEXITY SCORER\nMLM per-category\n0.0 (simple) \u2192 1.0 (hard)",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_classifier_to_complexity",
      "x": 370,
      "y": 310,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "classifier",
        "fixedPoint": [
          0.5,
          1
        ]
      },
      "endBinding": {
        "elementId": "complexity",
        "fixedPoint": [
          0.5,
          0
        ]
      }
    },
    {
      "type": "diamond",
      "id": "decision_diamond",
      "x": 530,
      "y": 245,
      "width": 100,
      "height": 80,
      "backgroundColor": "#fff3bf",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_decision_diamond",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_decision_diamond",
      "x": 540,
      "y": 255,
      "width": 80,
      "height": 60,
      "text": "DECISION\nTABLE\nsimple + det?",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "decision_diamond",
      "originalText": "DECISION\nTABLE\nsimple + det?",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_complexity_to_decision",
      "x": 460,
      "y": 390,
      "width": 170,
      "height": 105,
      "points": [
        [
          0,
          0
        ],
        [
          170,
          -105
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "complexity",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "decision_diamond",
        "fixedPoint": [
          0.3,
          1
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "kw_override",
      "x": 280,
      "y": 470,
      "width": 180,
      "height": 70,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#fff3bf",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_kw_override",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_kw_override",
      "x": 285,
      "y": 478,
      "width": 170,
      "height": 54,
      "text": "KEYWORD OVERRIDES\nsummariz* \u2192 summ\ndoc headers \u2192 summ\nNER patterns \u2192 ner\nsentiment shortcuts",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "kw_override",
      "originalText": "KEYWORD OVERRIDES\nsummariz* \u2192 summ\ndoc headers \u2192 summ\nNER patterns \u2192 ner\nsentiment shortcuts",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_classifier_to_kw",
      "x": 370,
      "y": 310,
      "width": 170,
      "height": 170,
      "points": [
        [
          0,
          0
        ],
        [
          -90,
          30
        ],
        [
          -90,
          170
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "classifier",
        "fixedPoint": [
          0,
          0.7
        ]
      },
      "endBinding": {
        "elementId": "kw_override",
        "fixedPoint": [
          1,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "det_solvers",
      "x": 530,
      "y": 370,
      "width": 180,
      "height": 80,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#d0bfff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_det_solvers",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_det_solvers",
      "x": 535,
      "y": 378,
      "width": 170,
      "height": 64,
      "text": "DETERMINISTIC SOLVERS\nArithmetic, Logic, NER,\nSentiment, Factual QA,\nCode Debug, Summarization",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "det_solvers",
      "originalText": "DETERMINISTIC SOLVERS\nArithmetic, Logic, NER,\nSentiment, Factual QA,\nCode Debug, Summarization",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_decision_to_det",
      "x": 580,
      "y": 325,
      "width": 0,
      "height": 45,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          45
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "decision_diamond",
        "fixedPoint": [
          0.5,
          1
        ]
      },
      "endBinding": {
        "elementId": "det_solvers",
        "fixedPoint": [
          0.5,
          0
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "det_output",
      "x": 700,
      "y": 390,
      "width": 130,
      "height": 50,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#b2f2bb",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_det_output",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_det_output",
      "x": 705,
      "y": 398,
      "width": 120,
      "height": 34,
      "text": "DET ANSWER\n(zero LLM cost)",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "det_output",
      "originalText": "DET ANSWER\n(zero LLM cost)",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_det_to_output",
      "x": 710,
      "y": 410,
      "width": 0,
      "height": 0,
      "points": [
        [
          0,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "strokeStyle": "dashed",
      "startBinding": {
        "elementId": "det_solvers",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "det_output",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "routing_table",
      "x": 530,
      "y": 490,
      "width": 180,
      "height": 70,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#fff3bf",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_routing_table",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_routing_table",
      "x": 535,
      "y": 498,
      "width": 170,
      "height": 54,
      "text": "ROUTING TABLE\n(GEPA override)\nmodel + prompt + role\nper category",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "routing_table",
      "originalText": "ROUTING TABLE\n(GEPA override)\nmodel + prompt + role\nper category",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_det_to_routing",
      "x": 530,
      "y": 430,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "det_solvers",
        "fixedPoint": [
          0.3,
          1
        ]
      },
      "endBinding": {
        "elementId": "routing_table",
        "fixedPoint": [
          0.3,
          0
        ]
      },
      "strokeStyle": "dashed"
    },
    {
      "type": "arrow",
      "id": "a_decision_to_routing",
      "x": 630,
      "y": 325,
      "width": 0,
      "height": 165,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          165
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "decision_diamond",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "routing_table",
        "fixedPoint": [
          1,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "local_llm",
      "x": 530,
      "y": 610,
      "width": 180,
      "height": 80,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#d0bfff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_local_llm",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_local_llm",
      "x": 535,
      "y": 618,
      "width": 170,
      "height": 64,
      "text": "LOCAL LLM INFERENCE\nGGUF (Qwen/Gemma/Qwen-Coder)\n1-3 models loaded\nConsensus voting (k\u22651)",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "local_llm",
      "originalText": "LOCAL LLM INFERENCE\nGGUF (Qwen/Gemma/Qwen-Coder)\n1-3 models loaded\nConsensus voting (k\u22651)",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_routing_to_local",
      "x": 615,
      "y": 560,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "routing_table",
        "fixedPoint": [
          0.5,
          1
        ]
      },
      "endBinding": {
        "elementId": "local_llm",
        "fixedPoint": [
          0.5,
          0
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "fireworks_api",
      "x": 530,
      "y": 740,
      "width": 180,
      "height": 70,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#ffd8a8",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_fireworks",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_fireworks",
      "x": 535,
      "y": 748,
      "width": 170,
      "height": 54,
      "text": "FIREWORKS API\n(Nemotron 70B /\nKimi-K2P7 /\nDeepSeek-v4 etc.)",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "fireworks_api",
      "originalText": "FIREWORKS API\n(Nemotron 70B /\nKimi-K2P7 /\nDeepSeek-v4 etc.)",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_local_to_fw",
      "x": 620,
      "y": 690,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "local_llm",
        "fixedPoint": [
          0.5,
          1
        ]
      },
      "endBinding": {
        "elementId": "fireworks_api",
        "fixedPoint": [
          0.5,
          0
        ]
      },
      "strokeStyle": "dashed"
    },
    {
      "type": "rectangle",
      "id": "workflow_engine",
      "x": 530,
      "y": 860,
      "width": 180,
      "height": 70,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#d0bfff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_workflow",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_workflow",
      "x": 535,
      "y": 868,
      "width": 170,
      "height": 54,
      "text": "WORKFLOW ENGINE\nPlan-and-Solve\nMath 3-step / Logic 3-step\nNER 2-step / Verify",
      "fontSize": 14,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "workflow_engine",
      "originalText": "WORKFLOW ENGINE\nPlan-and-Solve\nMath 3-step / Logic 3-step\nNER 2-step / Verify",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_fw_to_workflow",
      "x": 620,
      "y": 810,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "fireworks_api",
        "fixedPoint": [
          0.5,
          1
        ]
      },
      "endBinding": {
        "elementId": "workflow_engine",
        "fixedPoint": [
          0.5,
          0
        ]
      },
      "strokeStyle": "dashed"
    },
    {
      "type": "rectangle",
      "id": "qc_gate",
      "x": 780,
      "y": 610,
      "width": 150,
      "height": 80,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#ffc9c9",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_qc",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_qc",
      "x": 785,
      "y": 618,
      "width": 140,
      "height": 64,
      "text": "QC GATE\n(verify.py)\nHedge detection\nDegenerate check\nCode syntax (black+ruff)",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "qc_gate",
      "originalText": "QC GATE\n(verify.py)\nHedge detection\nDegenerate check\nCode syntax (black+ruff)",
      "autoResize": true
    },
    {
      "type": "rectangle",
      "id": "aggregator",
      "x": 780,
      "y": 740,
      "width": 150,
      "height": 70,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#c3fae8",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_aggregator",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_aggregator",
      "x": 785,
      "y": 748,
      "width": 140,
      "height": 54,
      "text": "AGGREGATION\n(Staging ReadyPool)\nMajority vote\nPer-type pools\nJudge selection",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "aggregator",
      "originalText": "AGGREGATION\n(Staging ReadyPool)\nMajority vote\nPer-type pools\nJudge selection",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_qc_to_agg",
      "x": 870,
      "y": 690,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "qc_gate",
        "fixedPoint": [
          0.6,
          1
        ]
      },
      "endBinding": {
        "elementId": "aggregator",
        "fixedPoint": [
          0.6,
          0
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "output",
      "x": 780,
      "y": 860,
      "width": 150,
      "height": 60,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#b2f2bb",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_output",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_output",
      "x": 785,
      "y": 870,
      "width": 140,
      "height": 40,
      "text": "OUTPUT\n/output/results.json\none answer per line",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "middle",
      "containerId": "output",
      "originalText": "OUTPUT\n/output/results.json\none answer per line",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_agg_to_output",
      "x": 870,
      "y": 810,
      "width": 0,
      "height": 50,
      "points": [
        [
          0,
          0
        ],
        [
          0,
          50
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "aggregator",
        "fixedPoint": [
          0.7,
          1
        ]
      },
      "endBinding": {
        "elementId": "output",
        "fixedPoint": [
          0.7,
          0
        ]
      }
    },
    {
      "type": "arrow",
      "id": "a_local_to_qc",
      "x": 710,
      "y": 650,
      "width": 70,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          70,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "local_llm",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "qc_gate",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "arrow",
      "id": "a_fw_to_qc",
      "x": 710,
      "y": 775,
      "width": 70,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          70,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "startBinding": {
        "elementId": "fireworks_api",
        "fixedPoint": [
          1,
          0.5
        ]
      },
      "endBinding": {
        "elementId": "aggregator",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "gepa_box",
      "x": 30,
      "y": 220,
      "width": 200,
      "height": 320,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#d0bfff",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_gepa_box",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_gepa_box",
      "x": 35,
      "y": 230,
      "width": 190,
      "height": 300,
      "text": "GEPA OPTIMIZATION\nFRAMEWORK\n\nOrchestrator run cycle:\n1. Seed Gen 0\n2. RUN LOOP:\n   a. Evaluate cells\n   b. Deduplicate\n   c. Pareto fronts\n   d. Analysis tags\n   e. Mutation + crossover\n   f. Repeat\n\n3. Publish routing\ntable to Pipeline\n\n4 objectives: accuracy,\nlatency, tokens, format",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "top",
      "containerId": "gepa_box",
      "originalText": "GEPA OPTIMIZATION\nFRAMEWORK\n\nOrchestrator run cycle:\n1. Seed Gen 0\n2. RUN LOOP:\n   a. Evaluate cells\n   b. Deduplicate\n   c. Pareto fronts\n   d. Analysis tags\n   e. Mutation + crossover\n   f. Repeat\n\n3. Publish routing\ntable to Pipeline\n\n4 objectives: accuracy,\nlatency, tokens, format",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_gepa_to_rt",
      "x": 230,
      "y": 500,
      "width": 300,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          300,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "strokeStyle": "dotted",
      "startBinding": {
        "elementId": "gepa_box",
        "fixedPoint": [
          1,
          0.6
        ]
      },
      "endBinding": {
        "elementId": "routing_table",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "staging_box",
      "x": 30,
      "y": 590,
      "width": 200,
      "height": 340,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#c3fae8",
      "fillStyle": "solid",
      "boundElements": [
        {
          "id": "t_staging_box",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_staging_box",
      "x": 35,
      "y": 600,
      "width": 190,
      "height": 320,
      "text": "STAGING PARALLEL\nSYSTEM (ReadyPool)\n\nPer-type partitioned\npools (det / local / fw)\n\nWorker processes:\n- Deterministic workers (2)\n- Local LLM workers (1-3)\n- Fireworks API workers (1)\n\nBackstop release after\nreservation_timeout_s\n\nPer-task judgment:\n- 5 votes per task\n- Majority wins\n- Judge selection\n\nHeartbeat monitoring\nOrphan task re-enqueue\nDeadline emergency\nAblation switches",
      "fontSize": 13,
      "fontFamily": 1,
      "strokeColor": "#1e1e1e",
      "textAlign": "center",
      "verticalAlign": "top",
      "containerId": "staging_box",
      "originalText": "STAGING PARALLEL\nSYSTEM (ReadyPool)\n\nPer-type partitioned\npools (det / local / fw)\n\nWorker processes:\n- Deterministic workers (2)\n- Local LLM workers (1-3)\n- Fireworks API workers (1)\n\nBackstop release after\nreservation_timeout_s\n\nPer-task judgment:\n- 5 votes per task\n- Majority wins\n- Judge selection\n\nHeartbeat monitoring\nOrphan task re-enqueue\nDeadline emergency\nAblation switches",
      "autoResize": true
    },
    {
      "type": "arrow",
      "id": "a_staging_to_agg",
      "x": 230,
      "y": 820,
      "width": 550,
      "height": 0,
      "points": [
        [
          0,
          0
        ],
        [
          550,
          0
        ]
      ],
      "endArrowhead": "arrow",
      "strokeStyle": "dotted",
      "startBinding": {
        "elementId": "staging_box",
        "fixedPoint": [
          1,
          0.55
        ]
      },
      "endBinding": {
        "elementId": "aggregator",
        "fixedPoint": [
          0,
          0.5
        ]
      }
    },
    {
      "type": "rectangle",
      "id": "legend",
      "x": 30,
      "y": 960,
      "width": 920,
      "height": 90,
      "roundness": {
        "type": 3
      },
      "backgroundColor": "#ffffff",
      "fillStyle": "solid",
      "strokeStyle": "dotted",
      "boundElements": [
        {
          "id": "t_legend",
          "type": "text"
        }
      ]
    },
    {
      "type": "text",
      "id": "t_legend",
      "x": 40,
      "y": 968,
      "width": 900,
      "height": 74,
      "text": "LEGEND\nBlue = Input/Classification  |  Green = Output  |  Yellow = Decision/Routing  |  Purple = Processing/LLM/Special  |  Orange = External API\nTeal = Storage/Data  |  Red = QC/Error  |  Dashed arrows = fallback/escalation path  |  Dotted arrows = feed from GEPA/Staging into main pipeline\nNAKED_CATEGORIES = {ner, summarization, factual, logic, math} \u2014 bypass system prompts, no deterministic solvers, go raw to local LLM",
      "fontSize": 12,
      "fontFamily": 1,
      "strokeColor": "#757575",
      "textAlign": "left",
      "verticalAlign": "top",
      "containerId": "legend",
      "originalText": "LEGEND\nBlue = Input/Classification  |  Green = Output  |  Yellow = Decision/Routing  |  Purple = Processing/LLM/Special  |  Orange = External API\nTeal = Storage/Data  |  Red = QC/Error  |  Dashed arrows = fallback/escalation path  |  Dotted arrows = feed from GEPA/Staging into main pipeline\nNAKED_CATEGORIES = {ner, summarization, factual, logic, math} \u2014 bypass system prompts, no deterministic solvers, go raw to local LLM",
      "autoResize": true
    }
  ],
  "appState": {
    "viewBackgroundColor": "#ffffff"
  }
}
```
%%
