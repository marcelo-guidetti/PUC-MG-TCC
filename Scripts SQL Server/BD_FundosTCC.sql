USE [master]
GO
/****** Object:  Database [Fundos_TCC]    Script Date: 10/09/2020 09:09:17 ******/
CREATE DATABASE [Fundos_TCC]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'Fundos_TCC', FILENAME = N'E:\Program Files\Microsoft SQL Server\MSSQL12.NOVAINSTANCIA2\MSSQL\DATA\Fundos_TCC.mdf' , SIZE = 2463744KB , MAXSIZE = UNLIMITED, FILEGROWTH = 1024KB )
 LOG ON 
( NAME = N'Fundos_TCC_log', FILENAME = N'E:\Program Files\Microsoft SQL Server\MSSQL12.NOVAINSTANCIA2\MSSQL\DATA\Fundos_TCC_log.ldf' , SIZE = 4224KB , MAXSIZE = 2048GB , FILEGROWTH = 10%)
GO
ALTER DATABASE [Fundos_TCC] SET COMPATIBILITY_LEVEL = 120
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [Fundos_TCC].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [Fundos_TCC] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [Fundos_TCC] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [Fundos_TCC] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [Fundos_TCC] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [Fundos_TCC] SET ARITHABORT OFF 
GO
ALTER DATABASE [Fundos_TCC] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [Fundos_TCC] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [Fundos_TCC] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [Fundos_TCC] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [Fundos_TCC] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [Fundos_TCC] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [Fundos_TCC] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [Fundos_TCC] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [Fundos_TCC] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [Fundos_TCC] SET  DISABLE_BROKER 
GO
ALTER DATABASE [Fundos_TCC] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [Fundos_TCC] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [Fundos_TCC] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [Fundos_TCC] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [Fundos_TCC] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [Fundos_TCC] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [Fundos_TCC] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [Fundos_TCC] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [Fundos_TCC] SET  MULTI_USER 
GO
ALTER DATABASE [Fundos_TCC] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [Fundos_TCC] SET DB_CHAINING OFF 
GO
ALTER DATABASE [Fundos_TCC] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [Fundos_TCC] SET TARGET_RECOVERY_TIME = 0 SECONDS 
GO
ALTER DATABASE [Fundos_TCC] SET DELAYED_DURABILITY = DISABLED 
GO
EXEC sys.sp_db_vardecimal_storage_format N'Fundos_TCC', N'ON'
GO
USE [Fundos_TCC]
GO
/****** Object:  Schema [Fundos_05052020]    Script Date: 10/09/2020 09:09:17 ******/
CREATE SCHEMA [Fundos_05052020]
GO
/****** Object:  Table [dbo].[indicesDiariosCDI]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[indicesDiariosCDI](
	[Ind_Data] [date] NOT NULL,
	[Ind_Rent] [float] NULL,
 CONSTRAINT [PK_indicesDiariosCDI] PRIMARY KEY CLUSTERED 
(
	[Ind_Data] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
/****** Object:  Table [dbo].[RentabilidadeAnualCDI]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RentabilidadeAnualCDI](
	[Ind_Data] [date] NOT NULL,
	[Ind_Rent] [float] NULL,
 CONSTRAINT [PK_RentabilidadeAnualCDI] PRIMARY KEY CLUSTERED 
(
	[Ind_Data] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
/****** Object:  Table [dbo].[RentabilidadeMensalCDI]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[RentabilidadeMensalCDI](
	[Ind_Data] [date] NOT NULL,
	[Ind_Rent] [float] NULL,
 CONSTRAINT [PK_RentabilidadeMensalCDI] PRIMARY KEY CLUSTERED 
(
	[Ind_Data] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO
/****** Object:  Table [Fundos_05052020].[administradores]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [Fundos_05052020].[administradores](
	[cnpj] [bigint] NOT NULL,
	[nome] [varchar](200) NULL,
	[slug] [varchar](100) NULL
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [Fundos_05052020].[Fundos]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [Fundos_05052020].[Fundos](
	[cnpj] [bigint] NOT NULL,
	[data] [date] NULL,
	[nome] [varchar](500) NULL,
	[patrimonio] [money] NULL,
	[cotistas] [int] NULL,
	[i] [bit] NULL,
	[t] [bit] NULL,
	[Tipo_Id] [int] NULL
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [Fundos_05052020].[fundos_dados]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [Fundos_05052020].[fundos_dados](
	[cnpj] [bigint] NOT NULL,
	[nome] [varchar](200) NULL,
	[nome_abreviado] [varchar](100) NULL,
	[de_cotas] [bit] NULL,
	[benchmark] [varchar](100) NULL,
	[cond_aberto] [bit] NULL,
	[prazo_pagamento] [varchar](100) NULL,
	[unique_slug] [varchar](100) NULL,
	[patrimonio] [money] NULL,
	[cotistas] [int] NULL,
	[last_update] [int] NULL,
	[last_quote_date] [int] NULL,
	[first_quote_date] [int] NULL,
	[exclusivo] [bit] NULL,
	[restrito] [bit] NULL,
	[invest_qualif] [bit] NULL,
	[tipo_de_previdencia] [bit] NULL,
	[tribut_lp] [bit] NULL,
	[situacao] [varchar](100) NULL,
	[status] [varchar](50) NULL,
	[tipo_de_fundo] [varchar](10) NULL,
	[indice] [varchar](10) NULL,
	[cnpjAdmin] [bigint] NULL,
	[classe_p_n] [varchar](100) NULL,
	[classe_p_s] [varchar](100) NULL,
	[classe_c_n] [varchar](100) NULL,
	[classe_c_s] [varchar](100) NULL
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [Fundos_05052020].[fundosXgestores]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Fundos_05052020].[fundosXgestores](
	[cnpjGestor] [bigint] NOT NULL,
	[cnpjFundo] [bigint] NOT NULL
) ON [PRIMARY]

GO
/****** Object:  Table [Fundos_05052020].[gestores]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [Fundos_05052020].[gestores](
	[cnpj] [bigint] NOT NULL,
	[nome] [varchar](200) NULL,
	[slug] [varchar](100) NULL
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  Table [Fundos_05052020].[indicesDiarios]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Fundos_05052020].[indicesDiarios](
	[cnpj] [bigint] NOT NULL,
	[data] [date] NOT NULL,
	[rentabilidade] [float] NULL,
	[patrimonio] [money] NULL,
	[q] [float] NULL
) ON [PRIMARY]

GO
/****** Object:  Table [Fundos_05052020].[rentabilidadeAnual]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Fundos_05052020].[rentabilidadeAnual](
	[cnpj] [bigint] NOT NULL,
	[data] [date] NOT NULL,
	[indice] [float] NULL,
	[patrimonio] [money] NULL,
	[q] [float] NULL,
	[rentabilidade] [float] NULL
) ON [PRIMARY]

GO
/****** Object:  Table [Fundos_05052020].[rentabilidadeMensal]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [Fundos_05052020].[rentabilidadeMensal](
	[cnpj] [bigint] NOT NULL,
	[data] [date] NOT NULL,
	[indice] [float] NULL,
	[patrimonio] [money] NULL,
	[q] [float] NULL,
	[rentabilidade] [float] NULL
) ON [PRIMARY]

GO
/****** Object:  Table [Fundos_05052020].[TBL_Tipos_Fundos]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
SET ANSI_PADDING ON
GO
CREATE TABLE [Fundos_05052020].[TBL_Tipos_Fundos](
	[Tipo_ID] [int] NOT NULL,
	[Tipo_SuperTipo] [varchar](50) NULL,
	[Tipo_Nome] [varchar](50) NULL,
	[Tipo_Aliq_IR] [float] NULL
) ON [PRIMARY]

GO
SET ANSI_PADDING OFF
GO
/****** Object:  View [dbo].[View_Fundos_Detalhes]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE VIEW [dbo].[View_Fundos_Detalhes]
AS
SELECT        Fundos_05052020.fundos_dados.*, Fundos_05052020.gestores.cnpj AS cnpjGestor, Fundos_05052020.gestores.nome AS nomeGestor, Fundos_05052020.administradores.nome AS nomeAdmin
FROM            Fundos_05052020.administradores INNER JOIN
                         Fundos_05052020.fundos_dados ON Fundos_05052020.administradores.cnpj = Fundos_05052020.fundos_dados.cnpjAdmin INNER JOIN
                         Fundos_05052020.gestores INNER JOIN
                         Fundos_05052020.fundosXgestores ON Fundos_05052020.gestores.cnpj = Fundos_05052020.fundosXgestores.cnpjGestor ON Fundos_05052020.fundos_dados.cnpj = Fundos_05052020.fundosXgestores.cnpjFundo

GO
/****** Object:  View [dbo].[View_Fundos_Mensal]    Script Date: 10/09/2020 09:09:17 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[View_Fundos_Mensal]
AS
SELECT        Fundos_05052020.Fundos.cnpj, Fundos_05052020.rentabilidadeMensal.data, Fundos_05052020.Fundos.nome, Fundos_05052020.Fundos.patrimonio, Fundos_05052020.Fundos.cotistas, Fundos_05052020.Fundos.i, 
                         Fundos_05052020.Fundos.t, Fundos_05052020.Fundos.Tipo_Id, Fundos_05052020.rentabilidadeMensal.rentabilidade, Fundos_05052020.TBL_Tipos_Fundos.Tipo_SuperTipo, 
                         Fundos_05052020.TBL_Tipos_Fundos.Tipo_Nome AS Expr1, Ind_Economicos.dbo.TBL_Ind_Rentabilidade_Mensal.Ind_Rent AS CDI, Fundos_05052020.fundos_dados.tipo_de_fundo
FROM            Fundos_05052020.rentabilidadeMensal INNER JOIN
                         Fundos_05052020.Fundos ON Fundos_05052020.rentabilidadeMensal.cnpj = Fundos_05052020.Fundos.cnpj INNER JOIN
                         Ind_Economicos.dbo.TBL_Ind_Rentabilidade_Mensal ON Fundos_05052020.rentabilidadeMensal.data = Ind_Economicos.dbo.TBL_Ind_Rentabilidade_Mensal.Ind_Data INNER JOIN
                         Fundos_05052020.TBL_Tipos_Fundos ON Fundos_05052020.Fundos.Tipo_Id = Fundos_05052020.TBL_Tipos_Fundos.Tipo_ID INNER JOIN
                         Fundos_05052020.fundos_dados ON Fundos_05052020.Fundos.cnpj = Fundos_05052020.fundos_dados.cnpj
WHERE        (Ind_Economicos.dbo.TBL_Ind_Rentabilidade_Mensal.Ind_Id = 1)


GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane1', @value=N'[0E232FF0-B466-11cf-A24F-00AA00A3EFFF, 1.00]
Begin DesignProperties = 
   Begin PaneConfigurations = 
      Begin PaneConfiguration = 0
         NumPanes = 4
         Configuration = "(H (1[40] 4[20] 2[20] 3) )"
      End
      Begin PaneConfiguration = 1
         NumPanes = 3
         Configuration = "(H (1 [50] 4 [25] 3))"
      End
      Begin PaneConfiguration = 2
         NumPanes = 3
         Configuration = "(H (1 [50] 2 [25] 3))"
      End
      Begin PaneConfiguration = 3
         NumPanes = 3
         Configuration = "(H (4 [30] 2 [40] 3))"
      End
      Begin PaneConfiguration = 4
         NumPanes = 2
         Configuration = "(H (1 [56] 3))"
      End
      Begin PaneConfiguration = 5
         NumPanes = 2
         Configuration = "(H (2 [66] 3))"
      End
      Begin PaneConfiguration = 6
         NumPanes = 2
         Configuration = "(H (4 [50] 3))"
      End
      Begin PaneConfiguration = 7
         NumPanes = 1
         Configuration = "(V (3))"
      End
      Begin PaneConfiguration = 8
         NumPanes = 3
         Configuration = "(H (1[56] 4[18] 2) )"
      End
      Begin PaneConfiguration = 9
         NumPanes = 2
         Configuration = "(H (1 [75] 4))"
      End
      Begin PaneConfiguration = 10
         NumPanes = 2
         Configuration = "(H (1[66] 2) )"
      End
      Begin PaneConfiguration = 11
         NumPanes = 2
         Configuration = "(H (4 [60] 2))"
      End
      Begin PaneConfiguration = 12
         NumPanes = 1
         Configuration = "(H (1) )"
      End
      Begin PaneConfiguration = 13
         NumPanes = 1
         Configuration = "(V (4))"
      End
      Begin PaneConfiguration = 14
         NumPanes = 1
         Configuration = "(V (2))"
      End
      ActivePaneConfig = 0
   End
   Begin DiagramPane = 
      Begin Origin = 
         Top = 0
         Left = 0
      End
      Begin Tables = 
         Begin Table = "fundos_dados (Fundos_05052020)"
            Begin Extent = 
               Top = 57
               Left = 32
               Bottom = 211
               Right = 226
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "administradores (Fundos_05052020)"
            Begin Extent = 
               Top = 146
               Left = 362
               Bottom = 259
               Right = 532
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "gestores (Fundos_05052020)"
            Begin Extent = 
               Top = 8
               Left = 729
               Bottom = 177
               Right = 899
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "fundosXgestores (Fundos_05052020)"
            Begin Extent = 
               Top = 25
               Left = 514
               Bottom = 121
               Right = 684
            End
            DisplayFlags = 280
            TopColumn = 0
         End
      End
   End
   Begin SQLPane = 
   End
   Begin DataPane = 
      Begin ParameterDefaults = ""
      End
   End
   Begin CriteriaPane = 
      Begin ColumnWidths = 11
         Column = 1440
         Alias = 2625
         Table = 4170
         Output = 2475
         Append = 1400
         NewValue = 1170
         SortType = 1350
         SortOrder = 1410
         GroupBy = 1350
         Filter = 1350
         Or = 1350
         Or = 1350
         Or = 1350
      End
   End
End
' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'View_Fundos_Detalhes'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane2', @value=N'410
         GroupBy = 1350
         Filter = 1350
         Or = 1350
         Or = 1350
         Or = 1350
      End
   End
End
' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'View_Fundos_Detalhes'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPaneCount', @value=1 , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'View_Fundos_Detalhes'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane1', @value=N'[0E232FF0-B466-11cf-A24F-00AA00A3EFFF, 1.00]
Begin DesignProperties = 
   Begin PaneConfigurations = 
      Begin PaneConfiguration = 0
         NumPanes = 4
         Configuration = "(H (1[40] 4[20] 2[20] 3) )"
      End
      Begin PaneConfiguration = 1
         NumPanes = 3
         Configuration = "(H (1 [50] 4 [25] 3))"
      End
      Begin PaneConfiguration = 2
         NumPanes = 3
         Configuration = "(H (1 [50] 2 [25] 3))"
      End
      Begin PaneConfiguration = 3
         NumPanes = 3
         Configuration = "(H (4 [30] 2 [40] 3))"
      End
      Begin PaneConfiguration = 4
         NumPanes = 2
         Configuration = "(H (1 [56] 3))"
      End
      Begin PaneConfiguration = 5
         NumPanes = 2
         Configuration = "(H (2 [66] 3))"
      End
      Begin PaneConfiguration = 6
         NumPanes = 2
         Configuration = "(H (4 [50] 3))"
      End
      Begin PaneConfiguration = 7
         NumPanes = 1
         Configuration = "(V (3))"
      End
      Begin PaneConfiguration = 8
         NumPanes = 3
         Configuration = "(H (1[56] 4[18] 2) )"
      End
      Begin PaneConfiguration = 9
         NumPanes = 2
         Configuration = "(H (1 [75] 4))"
      End
      Begin PaneConfiguration = 10
         NumPanes = 2
         Configuration = "(H (1[66] 2) )"
      End
      Begin PaneConfiguration = 11
         NumPanes = 2
         Configuration = "(H (4 [60] 2))"
      End
      Begin PaneConfiguration = 12
         NumPanes = 1
         Configuration = "(H (1) )"
      End
      Begin PaneConfiguration = 13
         NumPanes = 1
         Configuration = "(V (4))"
      End
      Begin PaneConfiguration = 14
         NumPanes = 1
         Configuration = "(V (2))"
      End
      ActivePaneConfig = 0
   End
   Begin DiagramPane = 
      Begin Origin = 
         Top = 0
         Left = 0
      End
      Begin Tables = 
         Begin Table = "rentabilidadeMensal (Fundos_05052020)"
            Begin Extent = 
               Top = 6
               Left = 38
               Bottom = 136
               Right = 208
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "Fundos (Fundos_05052020)"
            Begin Extent = 
               Top = 6
               Left = 246
               Bottom = 136
               Right = 416
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "TBL_Ind_Rentabilidade_Mensal (Ind_Economicos.dbo)"
            Begin Extent = 
               Top = 138
               Left = 38
               Bottom = 251
               Right = 208
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "TBL_Tipos_Fundos (Fundos_05052020)"
            Begin Extent = 
               Top = 138
               Left = 246
               Bottom = 268
               Right = 416
            End
            DisplayFlags = 280
            TopColumn = 0
         End
         Begin Table = "fundos_dados (Fundos_05052020)"
            Begin Extent = 
               Top = 6
               Left = 454
               Bottom = 136
               Right = 648
            End
            DisplayFlags = 280
            TopColumn = 23
         End
      End
   End
   Begin SQLPane = 
   End
   Begin DataPane = 
      Begin ParameterDefaults = ""
      End
      Begin ColumnWidths = 13
         Width = 284
         Width = 1500
         Width = 1500
         Width = 1500
         Width = 1500
         Width = 1500
         Width = 1500
         Width = 1500
         Width' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'View_Fundos_Mensal'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane2', @value=N' = 1500
         Width = 1500
         Width = 1500
         Width = 1500
         Width = 1500
      End
   End
   Begin CriteriaPane = 
      Begin ColumnWidths = 11
         Column = 1440
         Alias = 900
         Table = 1170
         Output = 720
         Append = 1400
         NewValue = 1170
         SortType = 1350
         SortOrder = 1410
         GroupBy = 1350
         Filter = 1350
         Or = 1350
         Or = 1350
         Or = 1350
      End
   End
End
' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'View_Fundos_Mensal'
GO
EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPaneCount', @value=2 , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'View_Fundos_Mensal'
GO
USE [master]
GO
ALTER DATABASE [Fundos_TCC] SET  READ_WRITE 
GO
