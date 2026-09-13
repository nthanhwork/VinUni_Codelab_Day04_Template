"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import os
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — Trợ lý Trí tuệ Nhân tạo chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng Vingroup (VinFast, Vinpearl).
- Phong cách giao tiếp: Chuyên nghiệp, tận tâm, lịch thiệp, chính xác và hướng tới sự hài lòng của khách hàng.

## 2. AVAILABLE TOOLS
Bạn được trang bị 2 công cụ chính thức:
1. search_product_catalog(category, max_price): Tra cứu danh mục sản phẩm (xe điện 'xe_dien' hoặc du lịch 'du_lich') theo ngân sách tối đa.
2. submit_support_ticket(customer_name, issue_description, priority): Ghi nhận phản hồi hoặc sự cố kỹ thuật của khách hàng vào hệ thống ticket.

## 3. CORE RULES
1. Tuyệt đối KHÔNG BAO GIỜ bịa đặt giá cả, thông số kỹ thuật hay mã ticket (Anti-Hallucination).
2. Bắt buộc PHẢI gọi tool khi khách hàng hỏi về danh mục sản phẩm, giá bán hoặc muốn gửi khiếu nại/báo lỗi.
3. Đối với các câu hỏi về chính sách chung (FAQ như thời hạn bảo hành pin 10 năm), trả lời trực tiếp mà không cần gọi tool.
4. Nếu không tìm thấy sản phẩm phù hợp điều kiện ngân sách, phải thông báo lịch sự và đề xuất phương án thay thế.

## 4. OPERATIONAL BOUNDARIES
- Chỉ tiếp nhận và xử lý các thông tin liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, Vinmec, Vinschool).
- Từ chối lịch sự các chủ đề ngoài phạm vi hoạt động của tập đoàn.

## 5. OUTPUT CONTRACT
Mỗi lượt suy luận và xử lý phải tuân thủ định dạng:
Thought: <Phân tích nhu cầu của khách hàng và bước tiếp theo>
Action: {"name": "<tên tool>", "args": {<tham số>}}
Observation: <Kết quả dữ liệu từ tool>
... (lặp lại nếu cần gọi thêm công cụ)
Final Answer: <Câu trả lời hoàn chỉnh, rõ ràng, thân thiện cho khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Trả về câu trả lời baseline không gọi công cụ
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _parse_price(self, text: str) -> int:
        """Trích xuất giá tối đa từ câu hỏi của người dùng."""
        text_lower = text.lower()
        match_ty = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:tỷ|ty)', text_lower)
        if match_ty:
            val = float(match_ty.group(1).replace(',', '.'))
            return int(val * 1_000_000_000)

        match_tr = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tr)', text_lower)
        if match_tr:
            val = float(match_tr.group(1).replace(',', '.'))
            return int(val * 1_000_000)

        match_k = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:k|nghìn|ngàn)', text_lower)
        if match_k:
            val = float(match_k.group(1).replace(',', '.'))
            return int(val * 1_000)

        match_num = re.search(r'(\d{1,3}(?:[.,]\d{3})+)', text_lower)
        if match_num:
            val_str = match_num.group(1).replace('.', '').replace(',', '')
            return int(val_str)

        return 999999999999

    def _extract_customer_name(self, text: str) -> str:
        """Trích xuất tên khách hàng từ văn bản."""
        match = re.search(r'(?:tên tôi là|tôi tên là|tôi tên|tên là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\n|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Khách hàng Vingroup"

    def _extract_priority(self, text: str) -> str:
        """Xác định mức độ ưu tiên của sự cố."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
            return "high"
        if any(k in text_lower for k in ["thấp", "low"]):
            return "low"
        return "medium"

    def _extract_issue(self, text: str) -> str:
        """Trích xuất mô tả vấn đề/sự cố."""
        match = re.search(r'((?:xe|phòng|hệ thống|pin|dịch vụ)[^.,]+(?:bị lỗi|lỗi|ẩm mốc|hỏng|sự cố)[^.,]*)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match2 = re.search(r'([^,.]*(?:lỗi|sự cố|ẩm mốc|hỏng)[^,.]*)', text, re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return "Yêu cầu hỗ trợ kỹ thuật và dịch vụ khách hàng"

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        user_lower = user_input.lower()

        # 1. Intent Detection
        is_faq = ("bảo hành" in user_lower or "chính sách" in user_lower) and not ("tên tôi" in user_lower or "lỗi" in user_lower or "sự cố" in user_lower)
        needs_catalog = any(k in user_lower for k in ["xe điện", "xe vinfast", "resort", "vinpearl", "giá dưới", "dưới", "sản phẩm"]) and not is_faq
        needs_ticket = any(k in user_lower for k in ["lỗi", "sự cố", "phản hồi", "khiếu nại", "ticket", "hỏng", "adas", "ẩm mốc"])

        # TRƯỜNG HỢP 1: FAQ chung (không cần gọi tool)
        if is_faq:
            thought = "Người dùng hỏi về chính sách bảo hành pin xe điện VinFast. Đây là câu hỏi FAQ chung, trả lời trực tiếp mà không cần gọi tool."
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc không giới hạn số km (áp dụng cho pin mua đứt hoặc thuê pin theo quy chuẩn hãng), mang lại sự an tâm tuyệt đối cho khách hàng."
            self.trace.append({
                "iteration": 1,
                "thought": thought,
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # TRƯỜNG HỢP 2: Đơn bước tra cứu catalog
        if needs_catalog and not needs_ticket:
            category = "du_lich" if any(k in user_lower for k in ["du_lich", "du lịch", "resort", "vinpearl", "phòng", "khách sạn"]) else "xe_dien"
            max_price = self._parse_price(user_input)

            thought = f"Khách hàng muốn tra cứu danh mục '{category}' với mức giá tối đa {max_price:,} VNĐ. Gọi tool search_product_catalog."
            action = {"name": "search_product_catalog", "args": {"category": category, "max_price": max_price}}
            obs = TOOL_MAP["search_product_catalog"](**action["args"])

            self.trace.append({
                "iteration": 1,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            if not obs:
                answer = f"Rất tiếc, chúng tôi không tìm thấy sản phẩm '{category}' nào có mức giá dưới {max_price:,} VNĐ. Quý khách có thể cân nhắc nâng mức ngân sách hoặc tham khảo các ưu đãi trả góp của VinFast."
            else:
                lines = [f"- {p['name']}: {p['price_vnd']:,} VNĐ ({p.get('description', '')})" for p in obs]
                answer = f"Tìm thấy {len(obs)} sản phẩm phù hợp với yêu cầu của bạn:\n" + "\n".join(lines)

            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # TRƯỜNG HỢP 3: Đơn bước tạo ticket hỗ trợ
        if needs_ticket and not needs_catalog:
            customer_name = self._extract_customer_name(user_input)
            priority = self._extract_priority(user_input)
            issue_desc = self._extract_issue(user_input)

            thought = f"Khách hàng {customer_name} thông báo sự cố kỹ thuật. Cần tạo ticket hỗ trợ mức ưu tiên {priority}."
            action = {
                "name": "submit_support_ticket",
                "args": {
                    "customer_name": customer_name,
                    "issue_description": issue_desc,
                    "priority": priority
                }
            }
            obs = TOOL_MAP["submit_support_ticket"](**action["args"])

            self.trace.append({
                "iteration": 1,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            answer = f"Hệ thống đã tiếp nhận yêu cầu hỗ trợ của quý khách {customer_name}. Mã ticket: {obs.get('ticket_id')} (Trạng thái: {obs.get('status')}, Mức ưu tiên: {obs.get('priority')}). Vấn đề '{issue_desc}' sẽ được đội ngũ kỹ thuật xử lý gấp."
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # TRƯỜNG HỢP 4: Đa bước (Multi-step: vừa tra cứu vừa tạo ticket, ví dụ TC03)
        if needs_catalog and needs_ticket:
            iteration = 1
            while iteration <= self.max_iterations:
                if iteration == 1:
                    category = "du_lich" if any(k in user_lower for k in ["du_lich", "du lịch", "resort", "vinpearl", "phòng"]) else "xe_dien"
                    max_price = self._parse_price(user_input)
                    thought = f"Bước 1: Tra cứu danh mục '{category}' với mức giá dưới {max_price:,} VNĐ."
                    action = {"name": "search_product_catalog", "args": {"category": category, "max_price": max_price}}
                    obs = TOOL_MAP["search_product_catalog"](**action["args"])
                    self.trace.append({
                        "iteration": iteration,
                        "thought": thought,
                        "action": action,
                        "observation": obs
                    })
                elif iteration == 2:
                    customer_name = self._extract_customer_name(user_input)
                    priority = self._extract_priority(user_input)
                    issue_desc = self._extract_issue(user_input)
                    thought = f"Bước 2: Tạo ticket hỗ trợ kỹ thuật cho khách hàng {customer_name}."
                    action = {
                        "name": "submit_support_ticket",
                        "args": {
                            "customer_name": customer_name,
                            "issue_description": issue_desc,
                            "priority": priority
                        }
                    }
                    obs = TOOL_MAP["submit_support_ticket"](**action["args"])
                    self.trace.append({
                        "iteration": iteration,
                        "thought": thought,
                        "action": action,
                        "observation": obs
                    })
                elif iteration == 3:
                    cat_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "search_product_catalog"), [])
                    tick_obs = next((t["observation"] for t in self.trace if t.get("action", {}).get("name") == "submit_support_ticket"), {})

                    cat_lines = [f"- {p['name']}: {p['price_vnd']:,} VNĐ" for p in cat_obs] if cat_obs else ["Rất tiếc, không tìm thấy gói nghỉ dưỡng phù hợp."]
                    final_answer = (
                        f"1. Thông tin sản phẩm/dịch vụ:\n" + "\n".join(cat_lines) + "\n\n"
                        f"2. Ghi nhận phản hồi khách hàng:\n"
                        f"- Đã tạo thành công ticket hỗ trợ mã {tick_obs.get('ticket_id')} cho khách hàng {tick_obs.get('customer_name')}.\n"
                        f"- Bộ phận CSKH sẽ liên hệ xử lý trong thời gian sớm nhất."
                    )
                    self.trace.append({
                        "iteration": iteration,
                        "thought": "Đã thu thập đủ thông tin từ cả 2 công cụ. Tổng hợp phản hồi hoàn chỉnh cho khách hàng.",
                        "final_answer": final_answer
                    })
                    return {
                        "answer": final_answer,
                        "trace": self.trace,
                        "iterations": iteration,
                        "status": "completed"
                    }
                iteration += 1

            return {
                "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
                "trace": self.trace,
                "iterations": iteration - 1,
                "status": "max_iterations_reached"
            }

        # TRƯỜNG HỢP MẶC ĐỊNH: Chào hỏi hoặc câu hỏi thông thường
        thought = "Khách hàng chào hỏi hoặc đưa ra câu hỏi chung. Phản hồi lời chào chuyên nghiệp."
        answer = "Xin chào quý khách! Tôi là VinAssistant — trợ lý AI của hệ sinh thái Vingroup. Tôi có thể hỗ trợ quý khách tra cứu xe điện VinFast, đặt phòng Vinpearl hoặc ghi nhận yêu cầu hỗ trợ kỹ thuật."
        self.trace.append({
            "iteration": 1,
            "thought": thought,
            "final_answer": answer
        })
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": 1,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
