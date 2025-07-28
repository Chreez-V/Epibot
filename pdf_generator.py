from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os

class EPI12PDFGenerator:
    def __init__(self):
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 20*mm
        
    def create_epi12_pdf(self, data, output_path, metadata):
        """Genera un PDF que replica exactamente la plantilla EPI-12 oficial"""
        
        # Crear el documento
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=8,
            fontName='Helvetica'
        )
        
        # Contenido del documento
        story = []
        
        # Encabezado oficial
        story.append(Paragraph("Gobierno Bolivariano de Venezuela", header_style))
        story.append(Paragraph("Ministerio del Poder Popular para la Salud", header_style))
        story.append(Paragraph("SIS", header_style))
        story.append(Spacer(1, 10))
        
        # Título del formato
        story.append(Paragraph("Formato SIS-04/EPI - 12", title_style))
        story.append(Paragraph("Consolidado Semanal de Enfermedades y Eventos de Notificación Obligatoria", title_style))
        story.append(Paragraph("Morbilidad", title_style))
        story.append(Spacer(1, 10))
        
        # Información del establecimiento - Arreglar el error aquí
        entidad = metadata.get('entidad', '') if metadata else ''
        municipio = metadata.get('municipio', '') if metadata else ''
        establecimiento = metadata.get('establecimiento', '') if metadata else ''
        año = metadata.get('año', datetime.now().year) if metadata else datetime.now().year
        semana = metadata.get('semana', '') if metadata else ''
        
        info_data = [
            ['Entidad:', entidad, 'Año:', str(año)],
            ['Municipio:', municipio, 'Semana:', str(semana)],
            ['Establecimiento:', establecimiento, '', ''],
        ]
        
        info_table = Table(info_data, colWidths=[25*mm, 50*mm, 20*mm, 30*mm])
        info_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 15))
        
        # Crear la tabla principal de enfermedades
        headers = [
            'Orden', 'Enfermedad / Evento',
            '< 1 año\nH    M',
            '1 a 4 años\nH    M', 
            '5 a 6 años\nH    M',
            '7 a 9 años\nH    M',
            '10 a 11 años\nH    M',
            '12 a 14 años\nH    M',
            '15 a 19 años\nH    M',
            '20 a 24 años\nH    M',
            '25 a 44 años\nH    M',
            '45 a 59 años\nH    M',
            '60 a 64 años\nH    M',
            '65 años y más\nH    M',
            'Edad Ignorada\nH    M',
            'Total\nHombres',
            'Total\nMujeres',
            'Total\nGeneral'
        ]
        
        # Datos de la tabla
        table_data = [headers]
        
        # Verificar que data tenga la estructura correcta
        enfermedades = data.get('enfermedades', []) if data else []
        
        for enfermedad in enfermedades:
            row = [
                str(enfermedad.get('orden', '')),
                enfermedad.get('enfermedad', ''),
                f"{enfermedad.get('menores_1_ano', {}).get('hombres', 0)}    {enfermedad.get('menores_1_ano', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_1_4_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_1_4_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_5_6_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_5_6_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_7_9_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_7_9_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_10_11_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_10_11_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_12_14_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_12_14_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_15_19_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_15_19_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_20_24_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_20_24_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_25_44_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_25_44_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_45_59_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_45_59_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('entre_60_64_anos', {}).get('hombres', 0)}    {enfermedad.get('entre_60_64_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('mayores_65_anos', {}).get('hombres', 0)}    {enfermedad.get('mayores_65_anos', {}).get('mujeres', 0)}",
                f"{enfermedad.get('edad_ignorada', {}).get('hombres', 0)}    {enfermedad.get('edad_ignorada', {}).get('mujeres', 0)}",
                str(enfermedad.get('total_hombres', 0)),
                str(enfermedad.get('total_mujeres', 0)),
                str(enfermedad.get('total_general', 0))
            ]
            table_data.append(row)
        
        # Crear la tabla con anchos de columna apropiados
        col_widths = [
            8*mm,   # Orden
            45*mm,  # Enfermedad
            12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm, 12*mm,  # Rangos de edad
            12*mm,  # Total Hombres
            12*mm,  # Total Mujeres
            12*mm   # Total General
        ]
        
        main_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        # Estilo de la tabla principal
        table_style = TableStyle([
            # Encabezados
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            
            # Datos
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # Columna orden
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),    # Columna enfermedad
            ('ALIGN', (2, 1), (-1, -1), 'CENTER'), # Resto de columnas
            ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),
            
            # Bordes
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
            
            # Resaltar totales
            ('BACKGROUND', (-3, 1), (-1, -1), colors.lightyellow),
        ])
        
        main_table.setStyle(table_style)
        story.append(main_table)
        
        # Pie de página
        story.append(Spacer(1, 20))
        story.append(Paragraph("Dirección General de Epidemiología", normal_style))
        story.append(Paragraph("Sistema de Información Epidemiológico Nacional", normal_style))
        story.append(Paragraph("Dirección de Vigilancia Epidemiológica", normal_style))
        story.append(Paragraph("Página 1 de 2", normal_style))
        
        # Generar el PDF
        doc.build(story)
        return output_path

def generar_pdf_epi12(document_data, metadata=None):
    """Genera un PDF EPI-12 y retorna la ruta del archivo"""
    if not metadata:
        metadata = {
            'entidad': 'Estado Bolívar',
            'municipio': 'Caroní',
            'establecimiento': 'Hospital General',
            'año': datetime.now().year,
            'semana': datetime.now().isocalendar()[1]
        }
    
    # Crear directorio si no existe
    os.makedirs("generated_pdfs", exist_ok=True)
    
    # Generar nombre único para el archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"EPI12_{timestamp}.pdf"
    output_path = os.path.join("generated_pdfs", filename)
    
    # Generar el PDF
    generator = EPI12PDFGenerator()
    generator.create_epi12_pdf(document_data, output_path, metadata)
    
    return output_path

