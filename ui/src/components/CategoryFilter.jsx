import { useState } from "react";
import cpiData from "../data/parsedCPIData_with_full_predictions.json";

const CategoryFilter = ({ selected, onChange, onReset }) => {
  // ✅ "CPI" 항목 제거
  const allCategories = Object.keys(cpiData).filter((cat) => cat !== "CPI"); // ★
  const [search, setSearch] = useState("");

  const toggleCategory = (category) => {
    if (selected.includes(category)) {
      onChange(selected.filter((c) => c !== category));
    } else {
      onChange([...selected, category]);
    }
  };

  const filteredCategories = allCategories.filter((cat) =>
    cat.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{
      border: "1px solid #d1d5db",
      borderRadius: "12px",
      padding: "1rem",
      backgroundColor: "#ffffff",
      maxHeight: "415px",
      minHeight: "280px",
      fontSize: "14px",
      display: "flex",
      flexDirection: "column",
      boxSizing: "border-box",
      width: "100%",
      maxWidth: "100%",
      overflow: "auto",
      flex: 1,               // ✅ 부모 높이만큼 차지
      height: "100%",
    }}>
      {/* 항목 슬라이서 + 검색창 */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        marginBottom: "0.75rem",
        flexWrap: "nowrap",
        width: "100%",
      }}>
        <div style={{
          fontWeight: "600",
          color: "#374151",
          whiteSpace: "nowrap",
          flexShrink: 0,
        }}>
          항목 슬라이서
        </div>

        <div style={{ flex: 1, position: "relative", maxWidth: "100%" }}>
          <input
            type="text"
            placeholder="항목을 검색하세요"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: "100%",
              maxWidth: "100%",
              minWidth: "0",
              boxSizing: "border-box",
              border: "1px solid #d1d5db",
              borderRadius: "6px",
              padding: "8px 12px",
              paddingLeft: "32px",
              fontSize: "13px",
              outline: "none",
              backgroundColor: "#f9fafb",
              transition: "all 0.2s ease",
            }}
          />
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            style={{
              position: "absolute",
              left: "10px",
              top: "50%",
              transform: "translateY(-50%)",
              width: "18px",
              height: "18px",
              color: "#9ca3af",
            }}
          >
            <path
              fill="currentColor"
              d="M15.5 14h-.79l-.28-.27a6.5 6.5 0 0 0 1.48-5.34c-.47-2.78-2.79-5-5.59-5.34a6.505 6.505 0 0 0-7.27 7.27c.34 2.8 2.56 5.12 5.34 5.59a6.5 6.5 0 0 0 5.34-1.48l.27.28v.79l4.25 4.25c.41.41 1.08.41 1.49 0 .41-.41.41-1.08 0-1.49L15.5 14zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"
            />
          </svg>
          {search && (
            <button
              onClick={() => setSearch("")}
              style={{
                position: "absolute",
                right: "10px",
                top: "50%",
                transform: "translateY(-50%)",
                background: "none",
                border: "none",
                cursor: "pointer",
                color: "#9ca3af",
                padding: "0",
              }}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* 항목 목록 */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
        paddingRight: "4px",
      }}>
        {filteredCategories.map((cat) => (
          <label key={cat} style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            color: "#111827",
            cursor: "pointer",
          }}>
            <input
              type="checkbox"
              checked={selected.includes(cat)}
              onChange={() => toggleCategory(cat)}
              style={{ accentColor: "#000000", width: "14px", height: "14px" }}
            />
            {cat}
          </label>
        ))}
        {filteredCategories.length === 0 && (
          <div style={{ fontSize: "13px", color: "#9ca3af" }}>
            일치하는 항목이 없습니다.
          </div>
        )}
      </div>

      {/* 초기화 버튼 */}
      <button
        onClick={onReset}
        style={{
          marginTop: "1rem",
          padding: "6px 10px",
          fontSize: "13px",
          border: "1px solid #d1d5db",
          backgroundColor: "#f3f4f6",
          borderRadius: "6px",
          cursor: "pointer",
          color: "#1f2937",
        }}
      >
        선택 초기화
      </button>
    </div>
  );
};

export default CategoryFilter;
