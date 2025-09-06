"""
MoldeAI Pro - Aplicativo de Conversão de Moldes com IA
Transforma fotos de moldes desenhados em PDFs técnicos profissionais
"""
import streamlit as st

# Carregar secrets de forma segura
def load_api_keys():
    try:
        vision_key = st.secrets["GOOGLE_VISION_API_KEY"]
        gemini_key = st.secrets["GEMINI_API_KEY"]
        return vision_key, gemini_key
    except:
        return None, None

# No main(), modificar para:
vision_api_key, gemini_api_key = load_api_keys()
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import fitz  # PyMuPDF
from google.cloud import vision
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
import tempfile
import os
from dataclasses import dataclass
from datetime import datetime
import math

# Configuração da página Streamlit
st.set_page_config(
    page_title="MoldeAI Pro",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para interface amigável
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        background-color: #FF6B6B;
        color: white;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #FF5252;
        transform: scale(1.02);
    }
    .main-header {
        text-align: center;
        color: #2C3E50;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class Piece:
    """Classe para representar uma peça do molde"""
    id: int
    name: str
    shape_type: str
    color: str
    contour: np.ndarray
    center: Tuple[int, int]
    dimensions: Dict[str, float]
    quantity: int = 1
    description: str = ""
    cutting_tips: str = ""

class MoldeAIProcessor:
    """Processador principal do MoldeAI Pro"""
    
    def __init__(self, vision_api_key: str = None, gemini_api_key: str = None):
        """
        Inicializa o processador com as APIs do Google
        
        Args:
            vision_api_key: Chave da API do Google Vision
            gemini_api_key: Chave da API do Gemini
        """
        self.vision_client = None
        self.gemini_model = None
        
        # Configurar Google Vision se a chave estiver disponível
        if vision_api_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = vision_api_key
            self.vision_client = vision.ImageAnnotatorClient()
        
        # Configurar Gemini se a chave estiver disponível
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
        
        # Parâmetros padrão
        self.grid_size_mm = 10  # 1cm por quadrado
        self.a4_width_mm = 210
        self.a4_height_mm = 297
        self.dpi = 300
        
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Corrige a perspectiva da imagem usando detecção de bordas
        
        Args:
            image: Imagem original
            
        Returns:
            Imagem com perspectiva corrigida
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar o maior contorno (provavelmente o papel)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Aproximar para um polígono
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:
                # Ordenar os pontos
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)
                
                # Calcular dimensões do retângulo de destino
                width = max(
                    np.linalg.norm(rect[0] - rect[1]),
                    np.linalg.norm(rect[2] - rect[3])
                )
                height = max(
                    np.linalg.norm(rect[0] - rect[3]),
                    np.linalg.norm(rect[1] - rect[2])
                )
                
                # Definir pontos de destino
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")
                
                # Calcular matriz de transformação
                M = cv2.getPerspectiveTransform(rect, dst)
                
                # Aplicar transformação
                warped = cv2.warpPerspective(image, M, (int(width), int(height)))
                return warped
        
        return image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordena os pontos no formato: top-left, top-right, bottom-right, bottom-left
        
        Args:
            pts: Array de pontos
            
        Returns:
            Pontos ordenados
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Soma e diferença para encontrar os pontos
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]     # Top-left
        rect[2] = pts[np.argmax(s)]     # Bottom-right
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect
    
    def detect_grid_scale(self, image: np.ndarray) -> float:
        """
        Detecta a escala da grade quadriculada
        
        Args:
            image: Imagem corrigida
            
        Returns:
            Escala em pixels por mm
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar linhas usando Hough Transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Calcular espaçamento médio entre linhas horizontais
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Linha aproximadamente horizontal
                    horizontal_lines.append(y1)
            
            if len(horizontal_lines) > 1:
                horizontal_lines.sort()
                spacings = np.diff(horizontal_lines)
                avg_spacing = np.median(spacings[spacings > 10])  # Filtrar ruído
                
                # Calcular pixels por mm (assumindo grade de 10mm)
                pixels_per_mm = avg_spacing / self.grid_size_mm
                return pixels_per_mm
        
        # Valor padrão se não conseguir detectar
        return 3.0  # ~3 pixels por mm
    
    def detect_shapes(self, image: np.ndarray, scale: float) -> List[Piece]:
        """
        Detecta formas geométricas na imagem
        
        Args:
            image: Imagem corrigida
            scale: Escala em pixels por mm
            
        Returns:
            Lista de peças detectadas
        """
        pieces = []
        
        # Pré-processamento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        piece_id = 1
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filtrar ruído
                continue
            
            # Aproximar forma
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Identificar tipo de forma
            num_vertices = len(approx)
            shape_type = self._identify_shape(num_vertices)
            
            # Calcular centro
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            
            # Detectar cor dominante
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(image, mask=mask)[:3]
            color = self._classify_color(mean_color)
            
            # Calcular dimensões
            x, y, w, h = cv2.boundingRect(contour)
            dimensions = {
                "width_mm": w / scale,
                "height_mm": h / scale,
                "area_mm2": area / (scale ** 2)
            }
            
            # Criar peça
            piece = Piece(
                id=piece_id,
                name=f"{shape_type} {piece_id}",
                shape_type=shape_type,
                color=color,
                contour=contour,
                center=(cx, cy),
                dimensions=dimensions
            )
            
            pieces.append(piece)
            piece_id += 1
        
        return pieces
    
    def _identify_shape(self, num_vertices: int) -> str:
        """
        Identifica o tipo de forma baseado no número de vértices
        
        Args:
            num_vertices: Número de vértices
            
        Returns:
            Nome da forma
        """
        if num_vertices == 3:
            return "Triângulo"
        elif num_vertices == 4:
            return "Quadrado"
        elif num_vertices == 5:
            return "Pentágono"
        elif num_vertices == 6:
            return "Hexágono"
        elif num_vertices > 6 and num_vertices <= 10:
            return "Estrela"
        else:
            return "Forma Irregular"
    
    def _classify_color(self, bgr_color: Tuple[float, float, float]) -> str:
        """
        Classifica a cor baseado em valores BGR
        
        Args:
            bgr_color: Tupla com valores BGR
            
        Returns:
            Nome da cor
        """
        b, g, r = bgr_color
        
        # Converter para HSV para melhor classificação
        rgb = np.uint8([[[r, g, b]]])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv
        
        # Classificar baseado em matiz
        if s < 50:
            if v < 85:
                return "Preto"
            elif v > 170:
                return "Branco"
            else:
                return "Cinza"
        elif h < 15 or h > 165:
            return "Vermelho"
        elif h < 35:
            return "Laranja"
        elif h < 45:
            return "Amarelo"
        elif h < 85:
            return "Verde"
        elif h < 125:
            return "Azul"
        else:
            return "Roxo"
    
    def extract_text_with_vision(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extrai texto usando Google Vision API
        
        Args:
            image: Imagem para análise
            
        Returns:
            Dicionário com textos detectados
        """
        if not self.vision_client:
            return {"texts": [], "numbers": []}
        
        # Converter imagem para formato da API
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            return {"texts": [], "numbers": []}
        
        image_content = encoded_image.tobytes()
        vision_image = vision.Image(content=image_content)
        
        # Detectar texto
        response = self.vision_client.document_text_detection(image=vision_image)
        
        texts = []
        numbers = []
        
        for text in response.text_annotations:
            content = text.description
            if content.isdigit():
                numbers.append({
                    "value": int(content),
                    "bounds": [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                })
            else:
                texts.append({
                    "content": content,
                    "bounds": [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                })
        
        return {"texts": texts, "numbers": numbers}
    
    def analyze_with_gemini(self, image: np.ndarray, pieces: List[Piece]) -> Dict[str, any]:
        """
        Analisa o molde usando Gemini para interpretação semântica
        
        Args:
            image: Imagem do molde
            pieces: Lista de peças detectadas
            
        Returns:
            Análise do Gemini
        """
        if not self.gemini_model:
            # Retornar análise padrão se Gemini não estiver configurado
            return self._default_analysis(pieces)
        
        # Preparar imagem para Gemini
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Criar prompt para análise
        prompt = f"""
        Analise este molde de costura e forneça:
        1. Nome sugestivo para cada uma das {len(pieces)} peças detectadas
        2. Quantidade recomendada de cada peça
        3. Dicas de corte e costura
        4. Identificação do padrão geral (quilt, patchwork, etc)
        5. Instruções de montagem
        
        Peças detectadas:
        """
        
        for piece in pieces:
            prompt += f"\n- Peça {piece.id}: {piece.shape_type} {piece.color}"
        
        # Enviar para análise
        response = self.gemini_model.generate_content([prompt, pil_image])
        
        # Processar resposta
        analysis = self._parse_gemini_response(response.text, pieces)
        
        return analysis
    
    def _default_analysis(self, pieces: List[Piece]) -> Dict[str, any]:
        """
        Análise padrão quando Gemini não está disponível
        
        Args:
            pieces: Lista de peças
            
        Returns:
            Análise padrão
        """
        analysis = {
            "pattern_type": "Molde Personalizado",
            "pieces_info": {}
        }
        
        for piece in pieces:
            analysis["pieces_info"][piece.id] = {
                "suggested_name": f"{piece.shape_type} {piece.color}",
                "quantity": 1,
                "cutting_tip": f"Cortar com 0.5cm de margem de costura",
                "description": f"{piece.shape_type} em tecido {piece.color.lower()}"
            }
        
        return analysis
    
    def _parse_gemini_response(self, response_text: str, pieces: List[Piece]) -> Dict[str, any]:
        """
        Processa a resposta do Gemini
        
        Args:
            response_text: Texto da resposta
            pieces: Lista de peças
            
        Returns:
            Análise estruturada
        """
        # Implementar parser para resposta do Gemini
        # Por enquanto, retornar análise padrão
        return self._default_analysis(pieces)
    
    def generate_pdf(self, pieces: List[Piece], analysis: Dict, scale: float) -> bytes:
        """
        Gera PDF técnico com moldes
        
        Args:
            pieces: Lista de peças
            analysis: Análise do Gemini
            scale: Escala em pixels por mm
            
        Returns:
            PDF em bytes
        """
        # Criar documento PDF
        doc = fitz.open()
        
        # Configurar página A4
        page_width = self.a4_width_mm * 2.83465  # mm para pontos
        page_height = self.a4_height_mm * 2.83465
        
        # Página 1: Grade e moldes
        page = doc.new_page(width=page_width, height=page_height)
        
        # Desenhar grade
        self._draw_grid(page)
        
        # Desenhar peças
        y_offset = 50
        x_offset = 50
        max_height = 0
        
        for piece in pieces:
            # Converter contorno para escala correta
            scaled_contour = piece.contour * (2.83465 / scale)  # Converter para pontos PDF
            
            # Verificar se cabe na linha atual
            bbox = cv2.boundingRect(piece.contour)
            piece_width = bbox[2] * (2.83465 / scale)
            piece_height = bbox[3] * (2.83465 / scale)
            
            if x_offset + piece_width > page_width - 50:
                # Nova linha
                x_offset = 50
                y_offset += max_height + 20
                max_height = 0
            
            if y_offset + piece_height > page_height - 50:
                # Nova página
                page = doc.new_page(width=page_width, height=page_height)
                self._draw_grid(page)
                y_offset = 50
                x_offset = 50
            
            # Desenhar peça
            self._draw_piece(page, piece, x_offset, y_offset, scale, analysis)
            
            x_offset += piece_width + 20
            max_height = max(max_height, piece_height)
        
        # Página 2: Legendas e instruções
        legend_page = doc.new_page(width=page_width, height=page_height)
        self._draw_legend(legend_page, pieces, analysis)
        
        # Salvar PDF
        pdf_bytes = doc.write()
        doc.close()
        
        return pdf_bytes
    
    def _draw_grid(self, page):
        """
        Desenha grade quadriculada na página
        
        Args:
            page: Página do PDF
        """
        # Configurar caneta para grade
        grid_color = (0.8, 0.8, 0.8)  # Cinza claro
        
        # Converter mm para pontos (1mm = 2.83465 pontos)
        grid_spacing = self.grid_size_mm * 2.83465
        
        # Desenhar linhas verticais
        x = grid_spacing
        while x < page.rect.width:
            page.draw_line(
                fitz.Point(x, 0),
                fitz.Point(x, page.rect.height),
                color=grid_color,
                width=0.5
            )
            x += grid_spacing
        
        # Desenhar linhas horizontais
        y = grid_spacing
        while y < page.rect.height:
            page.draw_line(
                fitz.Point(0, y),
                fitz.Point(page.rect.width, y),
                color=grid_color,
                width=0.5
            )
            y += grid_spacing
    
    def _draw_piece(self, page, piece: Piece, x_offset: float, y_offset: float, 
                    scale: float, analysis: Dict):
        """
        Desenha uma peça no PDF
        
        Args:
            page: Página do PDF
            piece: Peça a desenhar
            x_offset: Offset X
            y_offset: Offset Y
            scale: Escala
            analysis: Análise do Gemini
        """
        # Converter contorno para pontos do PDF
        points = []
        for point in piece.contour:
            px = point[0][0] * (2.83465 / scale) + x_offset
            py = point[0][1] * (2.83465 / scale) + y_offset
            points.append(fitz.Point(px, py))
        
        # Desenhar contorno
        shape = page.new_shape()
        shape.draw_polyline(points)
        shape.close_path()
        shape.finish(color=(0, 0, 0), width=1)
        shape.commit()
        
        # Adicionar número da peça
        piece_info = analysis.get("pieces_info", {}).get(piece.id, {})
        label = f"{piece.id}. {piece_info.get('suggested_name', piece.name)}"
        
        # Calcular posição do texto
        bbox = cv2.boundingRect(piece.contour)
        text_x = x_offset + (bbox[2] * (2.83465 / scale)) / 2
        text_y = y_offset + (bbox[3] * (2.83465 / scale)) / 2
        
        page.insert_text(
            fitz.Point(text_x, text_y),
            label,
            fontsize=10,
            color=(0, 0, 0)
        )
    
    def _draw_legend(self, page, pieces: List[Piece], analysis: Dict):
        """
        Desenha página de legendas
        
        Args:
            page: Página do PDF
            pieces: Lista de peças
            analysis: Análise do Gemini
        """
        # Título
        page.insert_text(
            fitz.Point(50, 50),
            "LEGENDAS E INSTRUÇÕES DE CORTE",
            fontsize=16,
            color=(0, 0, 0)
        )
        
        # Informações de cada peça
        y_pos = 100
        for piece in pieces:
            piece_info = analysis.get("pieces_info", {}).get(piece.id, {})
            
            text = f"""
Peça {piece.id}: {piece_info.get('suggested_name', piece.name)}
Forma: {piece.shape_type}
Cor: {piece.color}
Dimensões: {piece.dimensions['width_mm']:.1f}mm x {piece.dimensions['height_mm']:.1f}mm
Quantidade: {piece_info.get('quantity', 1)} unidade(s)
Dica de corte: {piece_info.get('cutting_tip', 'Cortar com margem de costura')}
            """
            
            page.insert_text(
                fitz.Point(50, y_pos),
                text,
                fontsize=10,
                color=(0, 0, 0)
            )
            
            y_pos += 100

# Interface Streamlit
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✂️ MoldeAI Pro</h1>
        <p>Transforme fotos de moldes em PDFs técnicos profissionais</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Chaves de API (em produção, usar secrets)
        st.markdown("""
        <div class="info-box">
        <b>APIs do Google</b><br>
        Configure suas chaves de API para habilitar todos os recursos.
        </div>
        """, unsafe_allow_html=True)
        
        vision_api_key = st.text_input("Google Vision API Key", type="password")
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        
        st.markdown("---")
        
        # Configurações de processamento
        st.subheader("📐 Parâmetros")
        grid_size = st.slider("Tamanho da grade (mm)", 5, 20, 10)
        
        st.markdown("---")
        
        # Informações
        st.info("""
        **Como usar:**
        1. Faça upload da foto do molde
        2. Clique em "Processar Molde"
        3. Baixe o PDF gerado
        """)
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload da Imagem")
        
        uploaded_file = st.file_uploader(
            "Escolha uma foto do molde",
            type=['jpg', 'jpeg', 'png'],
            help="Tire uma foto clara do molde em papel quadriculado"
        )
        
        if uploaded_file is not None:
            # Carregar imagem
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Exibir imagem original
            st.image(image, caption="Imagem Original", use_column_width=True)
            
            # Botão de processamento
            if st.button("🚀 Processar Molde", type="primary"):
                with st.spinner("Processando... Isso pode levar alguns segundos."):
                    # Criar processador
                    processor = MoldeAIProcessor(
                        vision_api_key=vision_api_key if vision_api_key else None,
                        gemini_api_key=gemini_api_key if gemini_api_key else None
                    )
                    processor.grid_size_mm = grid_size
                    
                    # Processar imagem
                    progress = st.progress(0)
                    
                    # Etapa 1: Correção de perspectiva
                    progress.progress(20, "Corrigindo perspectiva...")
                    corrected_image = processor.correct_perspective(image_np)
                    
                    # Etapa 2: Detectar escala
                    progress.progress(40, "Detectando escala da grade...")
                    scale = processor.detect_grid_scale(corrected_image)
                    
                    # Etapa 3: Detectar formas
                    progress.progress(60, "Identificando peças...")
                    pieces = processor.detect_shapes(corrected_image, scale)
                    
                    # Etapa 4: Análise com IA
                    progress.progress(80, "Analisando com IA...")
                    analysis = processor.analyze_with_gemini(corrected_image, pieces)
                    
                    # Etapa 5: Gerar PDF
                    progress.progress(90, "Gerando PDF...")
                    pdf_bytes = processor.generate_pdf(pieces, analysis, scale)
                    
                    progress.progress(100, "Concluído!")
                    
                    # Salvar resultados no session state
                    st.session_state['processed'] = True
                    st.session_state['corrected_image'] = corrected_image
                    st.session_state['pieces'] = pieces
                    st.session_state['analysis'] = analysis
                    st.session_state['pdf_bytes'] = pdf_bytes
                    st.session_state['scale'] = scale
    
    with col2:
        st.header("📊 Resultados")
        
        if 'processed' in st.session_state and st.session_state['processed']:
            # Exibir imagem corrigida
            st.image(
                st.session_state['corrected_image'],
                caption="Imagem Corrigida",
                use_column_width=True
            )
            
            # Informações das peças
            st.subheader(f"✂️ {len(st.session_state['pieces'])} peças detectadas")
            
            for piece in st.session_state['pieces']:
                piece_info = st.session_state['analysis'].get('pieces_info', {}).get(piece.id, {})
                with st.expander(f"Peça {piece.id}: {piece_info.get('suggested_name', piece.name)}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Forma:** {piece.shape_type}")
                        st.write(f"**Cor:** {piece.color}")
                    with col_b:
                        st.write(f"**Largura:** {piece.dimensions['width_mm']:.1f}mm")
                        st.write(f"**Altura:** {piece.dimensions['height_mm']:.1f}mm")
                    
                    st.write(f"**Quantidade sugerida:** {piece_info.get('quantity', 1)}")
                    st.write(f"**Dica de corte:** {piece_info.get('cutting_tip', 'Cortar com margem de costura')}")
            
            # Botão de download do PDF
            st.markdown("---")
            st.subheader("📥 Download do PDF")
            
            # Criar botão de download
            pdf_b64 = base64.b64encode(st.session_state['pdf_bytes']).decode()
            href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="moldes_gerados_v1.pdf">📄 Baixar Molde Final (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Informações adicionais
            st.success("✅ Processamento concluído com sucesso!")
            
            st.markdown("""
            <div class="info-box">
            <b>PDF Gerado com:</b><br>
            • Grade quadriculada de {0}mm<br>
            • Escala 1:1 para impressão<br>
            • Página de legendas com instruções<br>
            • Formato A4 pronto para imprimir
            </div>
            """.format(grid_size), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            <b>Aguardando processamento...</b><br>
            Faça upload de uma imagem e clique em "Processar Molde" para começar.
            </div>
            """, unsafe_allow_html=True)
    
    # Footer com instruções
    st.markdown("---")
    
    with st.expander("📖 Instruções Detalhadas"):
        st.markdown("""
        ### Como tirar uma boa foto do molde:
        
        1. **Iluminação**: Use luz natural ou ambiente bem iluminado
        2. **Ângulo**: Posicione o celular paralelo ao papel
        3. **Enquadramento**: Capture todo o papel na foto
        4. **Foco**: Certifique-se que a imagem está nítida
        5. **Grade visível**: A grade quadriculada deve estar clara
        
        ### O que o sistema faz automaticamente:
        
        ✅ **Correção de perspectiva** - Ajusta automaticamente fotos inclinadas  
        ✅ **Detecção de escala** - Calcula o tamanho real baseado na grade  
        ✅ **Identificação de formas** - Reconhece triângulos, quadrados, estrelas, etc  
        ✅ **Leitura de números** - Identifica numeração das peças  
        ✅ **Análise inteligente** - Sugere quantidades e dicas de corte  
        ✅ **PDF profissional** - Gera documento técnico pronto para imprimir  
        
        ### Dicas para melhor resultado:
        
        - Use papel quadriculado padrão (1cm x 1cm)
        - Desenhe com caneta preta ou azul escura
        - Numere cada peça claramente
        - Use cores diferentes para peças diferentes
        """)
    
    with st.expander("🤖 Sobre a Tecnologia"):
        st.markdown("""
        ### MoldeAI Pro utiliza:
        
        **Google Vision AI** 🔍
        - Detecção avançada de formas e contornos
        - OCR para leitura de números e textos
        - Análise de cores e padrões
        
        **Google Gemini** 🧠
        - Interpretação semântica do design
        - Sugestões inteligentes de corte
        - Identificação de padrões de costura
        
        **OpenCV** 📐
        - Correção automática de perspectiva
        - Processamento de imagem
        - Detecção de grade quadriculada
        
        **PyMuPDF** 📄
        - Geração de PDF vetorial
        - Grade milimetrada precisa
        - Escala 1:1 para impressão
        """)
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>MoldeAI Pro v1.0 | Desenvolvido com ❤️ para artesãs e costureiras</p>
        <p style="font-size: 12px;">© 2024 - Tecnologia a serviço da arte manual</p>
    </div>
    """, unsafe_allow_html=True)

# Executar aplicação
if __name__ == "__main__":
    # Configurar session state
    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
    
    # Executar interface principal
    main()
