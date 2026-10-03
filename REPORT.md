# Báo cáo Day 17: Memory Systems for AI Agent

## 1. Cách chạy

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest

python src/benchmark.py           # 2 bảng: Standard + Long-Context Stress
pytest src/test_agents.py -v      # 5 test
```

Mặc định cả hai agent chạy **offline, deterministic**, không cần API key. Khi có `.env` với `LLM_PROVIDER`, `LLM_MODEL` và key tương ứng, agent tự chuyển sang chế độ live (LangChain `create_agent`). Hỗ trợ 6 provider: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`. Trên Windows nên đặt `PYTHONIOENCODING=utf-8` để in tiếng Việt.

Biến cấu hình compact: `COMPACT_THRESHOLD_TOKENS` (mặc định 600) và `COMPACT_KEEP_MESSAGES` (mặc định 4).

## 2. Thiết kế: tách bạch ba lớp memory

| Lớp | Nơi lưu | Vòng đời | Chứa gì |
|---|---|---|---|
| **Short-term** | `CompactMemoryManager.state[thread_id]["messages"]` (live: `InMemorySaver`) | Một thread | Các message gần nhất, nguyên văn |
| **Persistent** | `state/profiles/<user>/User.md` | Qua mọi thread, qua cả lần khởi động lại process | Fact ổn định: tên, nơi ở, nghề, style, đồ uống, món ăn, thú cưng, mối quan tâm |
| **Compact** | `CompactMemoryManager.state[thread_id]["summary"]` (live: `SummarizationMiddleware`) | Một thread | Tóm tắt có giới hạn (≤ 8 dòng) của các message cũ đã bị nén |

Một lượt của Advanced Agent:

```
message → extract_profile_updates() → upsert_fact() vào User.md
        → CompactMemoryManager.append()   (tự compact khi vượt ngưỡng)
        → prompt = User.md + summary + recent messages
        → trả lời → cập nhật bộ đếm token
```

**So sánh công bằng.** Hai agent dùng chung một hàm sinh câu trả lời `answer_from_facts()`. Khác biệt duy nhất là **nguồn fact**: Baseline chỉ có fact nghe được trong thread hiện tại (`SessionState.facts`, mất khi sang thread mới), còn Advanced đọc từ `User.md`. Vì vậy chênh lệch trong benchmark phản ánh đúng memory system chứ không phải chất lượng "lời văn".

**Tính token.** `estimate_tokens` ≈ số ký tự / 4, dùng chung cho cả hai agent.
- `Agent tokens only`: tổng token của message người dùng và câu trả lời.
- `Prompt tokens processed`: tổng ngữ cảnh đưa vào model qua các lượt. Baseline dùng **toàn bộ lịch sử thread**. Advanced dùng `User.md` + summary + các message được giữ lại.

## 3. Kết quả benchmark

Recall được hỏi trong **thread mới, ngay sau mỗi hội thoại**, nên phản ánh đúng trạng thái memory tại thời điểm đó (ví dụ sau conv-03, nơi ở phải là Huế). Mỗi lần chạy, benchmark dùng thư mục `state/benchmark/` mới để `User.md` cũ không làm sai lệch kết quả.

### Standard Benchmark (`conversations.json`: 10 hội thoại, 14 câu recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2254 | 10175 | 0% | 0.15 | 0 | 0 |
| Advanced | 2364 | 18024 (**+77%**) | **100%** | 1.00 | 350 | 0 |

### Long-Context Stress Benchmark (`advanced_long_context.json`: 16 lượt dài, 3 câu recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2533 | 21148 | 0% | 0.15 | 0 | 0 |
| Advanced | 2589 | 9062 (**−57%**) | **100%** | 1.00 | 253 | 6 |

Prompt tokens theo từng lượt trên thread stress:

| Lượt | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 187 | 354 | 507 | 667 | 988 | 1283 | 1523 | 1823 | 2100 | 2388 |
| Advanced | 223 | 390 | 543 | **379** | 427 | 468 | 709 | 654 | 507 | 795 |
| Số lần compact (cộng dồn) | 0 | 0 | 0 | 1 | 2 | 3 | 3 | 4 | 5 | 6 |

`Response quality` là thước đo heuristic: 70% tỷ lệ fact đúng, 15% ngắn gọn (≤ 400 ký tự), 15% trình bày dạng bullet, theo đúng style người dùng yêu cầu trong data.

## 4. Phân tích

### 4.1. Vì sao Advanced có recall tốt hơn Baseline

Baseline chỉ có within-session memory. Khi câu hỏi recall đến ở thread mới, `SessionState` của thread đó rỗng nên nó trả lời "chưa có thông tin" (recall 0%). Đây là hành vi **đúng như thiết kế**, không phải lỗi. Advanced ghi fact ổn định vào `User.md` trên đĩa, nên mọi thread mới, thậm chí một instance agent mới (đã có test kiểm tra), đều đọc lại được.

Recall 100% còn nhờ xử lý đúng **correction**: Đà Nẵng → Huế (conv-03), backend → MLOps engineer (conv-06), Huế → Đà Nẵng (stress). `upsert_fact` **ghi đè** fact cũ thay vì thêm dòng mới, nên `User.md` không bao giờ chứa đồng thời fact cũ và fact mới.

### 4.2. Vì sao Advanced tốn hơn ở hội thoại ngắn (+77% prompt)

Mỗi lượt của Advanced phải mang theo `User.md` (khoảng 55–85 token) vào prompt, kể cả những lượt chỉ là "Đã ghi nhận." Hội thoại standard ngắn (10 lượt, khoảng 15 token mỗi lượt), lịch sử của Baseline chưa kịp phình, và **không lần nào chạm ngưỡng compact (0 compaction)**. Với khoảng 124 lượt trên toàn bộ suite, chi phí cố định của profile cộng lại thành khoảng +7.800 prompt token.

Nói cách khác, persistent memory có **chi phí cố định mỗi lượt**, và chi phí này chỉ được bù khi lịch sử thread đủ dài. Ở lượt 1–3 của bảng stress cũng thấy rõ: Advanced đắt hơn Baseline đúng bằng kích thước `User.md` (+36 token) cho đến lần compact đầu tiên.

### 4.3. Vì sao compact giúp Advanced thắng ở hội thoại dài (−57% prompt)

Prompt của Baseline mỗi lượt bằng toàn bộ lịch sử, nên **tăng tuyến tính theo số lượt** (187 → 2388), và tổng cộng dồn tăng **bậc hai**. Prompt của Advanced bị **chặn trên**: khi các message giữ lại vượt 600 token, các message cũ được gộp vào summary (≤ 8 dòng), nên prompt dao động theo hình răng cưa trong khoảng 380–800 token bất kể thread dài đến đâu. Điểm giao nhau là **lượt 4**, ngay khi compact lần đầu xảy ra. Thread càng dài thì khoảng cách càng lớn.

Compact tối ưu **`Prompt tokens processed`**, không tối ưu **`Agent tokens only`**. Hai agent nhận cùng message và trả lời gần giống nhau, nên agent tokens chỉ chênh khoảng 2% (2533 so với 2589). Phần tiết kiệm đến hoàn toàn từ việc không phải gửi lại lịch sử cũ ở mỗi lượt. Test `test_compact_reduces_prompt_load_on_long_thread` kiểm tra cả hai vế: prompt giảm hơn 40%, agent tokens chênh không quá 5%.

**Cái giá của compact.** Summary heuristic chỉ giữ câu đầu (≤ 120 ký tự) của mỗi message người dùng. Phần lớn chi tiết tin tức trong stress test (số liệu WMO, độ cao X-59, …) bị mất. Điều này chấp nhận được **chỉ vì** fact ổn định đã nằm trong `User.md` chứ không phụ thuộc vào summary. Nếu người dùng hỏi follow-up về chi tiết đã bị nén, Advanced sẽ trả lời kém hơn Baseline. Ở chế độ live, `SummarizationMiddleware` dùng LLM để tóm tắt nên giữ được nhiều ý hơn, nhưng mỗi lần compact tốn thêm một lần gọi model.

### 4.4. File memory tăng trưởng thế nào và rủi ro đi kèm

`User.md` tăng lên 350 byte sau 10 phiên (8 fact) và 253 byte sau thread stress (5 fact). File tăng **theo số fact khác nhau, không theo số lượt**, vì correction ghi đè thay vì thêm dòng. Riêng `response_style` và `interests` được **gộp dồn** (union) qua các phiên, nên đây là hai trường có thể phình mãi: `response_style` của `dungct` đã có 6 trait, trong đó có các trait gần trùng nhau ("ví dụ thực tế" và "ví dụ thực chiến", "bullet" và "3 bullet").

Các rủi ro chính:
- **Phình to**: `User.md` nằm trong prompt của *mọi* lượt, nên mỗi byte thêm vào nhân lên theo số lượt (chính là nguyên nhân của +77% ở mục 4.2). Cần memory decay hoặc giới hạn số trait.
- **Lưu sai fact**: extractor ghi sai thì sai đó tồn tại vĩnh viễn và xuất hiện ở mọi phiên sau. Lỗi này nguy hiểm hơn lỗi của short-term memory, vốn tự mất khi hết thread.
- **Quyền riêng tư**: `User.md` là dữ liệu cá nhân dạng plain text trên đĩa. Production cần cơ chế xem, sửa và xóa theo yêu cầu người dùng.

## 5. Bonus đã làm

### 5.1. Conflict handling

**Vấn đề:** người dùng đính chính (đổi nơi ở, đổi nghề); agent ngây thơ sẽ giữ cả fact cũ lẫn fact mới, hoặc lấy mẩu text mới nhất một cách máy móc.

**Cách làm:**
- `upsert_fact` thay thế đúng dòng `- key: value` cũ, nên mỗi key luôn chỉ có một giá trị.
- Trong một message, `_last_affirmed` chọn **giá trị được khẳng định cuối cùng**. Ví dụ "Lúc đầu nói Huế, nhưng thực ra … ở Đà Nẵng", hay "từ Huế sang Đà Nẵng", đều cho ra Đà Nẵng.
- Các giá trị bị phủ định ngay trước đó bị bỏ qua: "không còn ở Đà Nẵng", "không còn là backend engineer".

**Kết quả:** tất cả câu hỏi về nơi ở và nghề trong cả hai bộ data đều trả về giá trị mới nhất. `test_cross_session_recall` kiểm tra câu trả lời **không** chứa "Đà Nẵng" hay "backend" sau khi đã đính chính.

### 5.2. Guardrail trước khi ghi vào `User.md` (dạng đơn giản của confidence threshold)

**Vấn đề:** các câu hỏi và câu nói *về* một fact (không phải *khẳng định* fact) dễ bị ghi nhầm vào memory.

**Cách làm:** `extract_profile_updates` xét từng câu và chỉ ghi khi câu đủ "chắc chắn":
- câu hỏi (kết thúc bằng `?`) không bao giờ ghi, ví dụ "Bạn có biết DũngCT không?";
- câu có dấu hiệu nhiễu ("đùa", "ví dụ cũ") bị bỏ qua, ví dụ "đùa … chuyển sang product manager", "nhắc lại Đà Nẵng như ví dụ cũ";
- "Hà Nội chỉ là nơi mình bay ra họp" không khớp pattern nơi ở, vì pattern yêu cầu dạng `ở|sang|là <thành phố>`.

**Kết quả:** stress test có 3 nguồn nhiễu (Huế cũ, Hà Nội, product manager), và cả 3 đều không lọt vào `User.md`. Có test riêng: `test_extractor_ignores_questions_noise_and_negation`.

### 5.3. Rủi ro của các bonus này

- Guardrail là **luật cứng**, không phải điểm tin cậy thật. Nó có thể chặn nhầm fact thật, ví dụ "Mình ở Huế, bạn biết chỗ nào ngon không?" bị bỏ vì kết thúc bằng `?`, nên recall giảm. Ngược lại, câu đùa không chứa từ "đùa" vẫn lọt qua.
- Ghi đè theo key nghĩa là **mất lịch sử**: không trả lời được "trước đây mình ở đâu?". Nếu cần, nên lưu thêm phần lịch sử kèm timestamp, chỉ đưa giá trị hiện tại vào prompt.
- Extractor dùng regex và danh sách từ khóa cố định (thành phố, trait style, interest) **được thiết kế theo bộ data này**. Recall 100% là trần trên ở điều kiện thuận lợi; dữ liệu thật sẽ thấp hơn. Hướng nâng cấp là dùng LLM hoặc NER để trích fact có cấu trúc kèm điểm confidence, rồi đặt ngưỡng trước khi `upsert_fact`. Ở chế độ live, tool `save_user_fact` đã cho LLM tự quyết định, còn extractor deterministic đóng vai lưới an toàn.

## 6. Kết luận

1. Baseline không nhớ dài hạn: recall 0% ở thread mới.
2. Advanced thêm `User.md` nên recall lên 100% và giữ đúng fact sau correction.
3. Hội thoại dài làm prompt của Baseline tăng tuyến tính mỗi lượt (187 → 2388 token).
4. Compact memory chặn prompt của Advanced quanh 380–800 token, giảm 57% tổng prompt; agent tokens gần như không đổi.
5. Cái giá: hội thoại ngắn tốn hơn 77% prompt, summary làm mất chi tiết, và `User.md` cần guardrail (conflict handling, chặn câu hỏi và nhiễu) để không lưu sai. Hệ thống mạnh hơn nhưng cũng phức tạp hơn.
